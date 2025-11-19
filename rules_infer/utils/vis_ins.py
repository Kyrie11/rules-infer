# visualize_events_optimized.py (with view filtering)

import os
import json
import time
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# --- 配置区 ---
NUSCENES_DATAROOT = '/data0/senzeyu2/dataset/nuscenes'
NUSCENES_VERSION = 'v1.0-trainval'
EVENTS_JSON_PATH = 'social_events.json'
OUTPUT_DIR = '/data0/senzeyu2/dataset/nuscenes/events_ins'  # 新的输出目录
PRIMARY_AGENT_COLOR = (1, 0, 0)  # Red
INTERACTING_AGENT_COLOR = (0, 0, 1)  # Blue


# --- 核心投影与绘制函数 (保持不变, 但修复一个小bug) ---
def project_box_to_image(box, camera_intrinsic, ego_pose, cam_pose):
    corners_3d = box.corners()
    ego_pos = np.array(ego_pose['translation'])
    ego_rot = Quaternion(ego_pose['rotation'])
    corners_3d = corners_3d - ego_pos[:, np.newaxis]
    corners_3d = np.dot(ego_rot.inverse.rotation_matrix, corners_3d)
    cam_pos = np.array(cam_pose['translation'])
    cam_rot = Quaternion(cam_pose['rotation'])
    corners_3d = corners_3d - cam_pos[:, np.newaxis]
    corners_3d = np.dot(cam_rot.inverse.rotation_matrix, corners_3d)
    depth = corners_3d[2, :]
    # ### BUG FIX ###: 使用np.any，只要有一个角点在相机后面，投影就无效
    # if np.any(depth <= 0.1):
    if np.all(depth<=0.1):
        return None
    points_2d = np.dot(camera_intrinsic, corners_3d)
    points_2d[:2, :] = points_2d[:2, :] / points_2d[2, :]
    return points_2d.T


def draw_projected_box(ax, points_2d, color, linewidth, label=None):
    # (此函数无需修改)
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for edge in edges:
        start_point, end_point = points_2d[edge[0]], points_2d[edge[1]]
        ax.plot([start_point[0], end_point[0]],     [start_point[1], end_point[1]], color=color, linewidth=linewidth)

    if label is not None:
        x, y, z= points_2d[0]
        ax.text(x, y, str(label), color=color, fontsize=8, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

# --- 辅助函数 (保持不变) ---
def get_annotation_for_instance(nusc, sample, instance_token):
    # (此函数无需修改)
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token:
            return ann
    return None


def get_sample_tokens_by_frame_range(nusc, scene, start_frame, end_frame):
    # (此函数无需修改)
    sample_tokens = []
    current_token = scene['first_sample_token']
    for _ in range(start_frame):
        if not current_token: return []
        current_token = nusc.get('sample', current_token)['next']
    for _ in range(start_frame, end_frame + 1):
        if not current_token: break
        sample_tokens.append(current_token)
        current_token = nusc.get('sample', current_token)['next']
    return sample_tokens


# --- ### CORE REWRITE ###: 带有视角筛选功能的可视化主函数 ---
def visualize_event_with_filtering(nusc, event_data, base_output_dir):
    """
    为单个事件可视化，但只保存包含关键或交互Agent的摄像头视角，
    并为VLM生成一个manifest.json文件。
    """
    # 1. 提取信息并创建输出目录
    scene_token = event_data['scene_token']
    key_agent_token = event_data['key_agent_token']
    interacting_agent_tokens = event_data['interacting_agent_tokens']
    start_frame, end_frame = event_data['event_start_frame'], event_data['event_end_frame']

    event_id = f"{scene_token}_{key_agent_token}_{start_frame}-{end_frame}"
    event_output_dir = os.path.join(base_output_dir, event_id)
    os.makedirs(event_output_dir, exist_ok=True)

    # 2. 获取事件时间范围内的所有样本
    scene_record = nusc.get('scene', scene_token)
    samples_in_event = get_sample_tokens_by_frame_range(nusc, scene_record, start_frame, end_frame)
    if not samples_in_event: return None, 0

    # 3. 初始化为VLM准备的清单 (Manifest)
    event_manifest = {"event_id": event_id, "frames": {}}

    cam_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']

    # 4. 遍历事件中的每一帧
    for relative_frame_idx, sample_token in enumerate(samples_in_event):
        sample = nusc.get('sample', sample_token)

        # 收集本帧所有需要关注的agent标注
        target_anns = []
        primary_ann = get_annotation_for_instance(nusc, sample, key_agent_token)
        if primary_ann: target_anns.append(primary_ann)
        for token in interacting_agent_tokens:
            ann = get_annotation_for_instance(nusc, sample, token)
            if ann: target_anns.append(ann)

        if not target_anns:  # 如果这一帧，我们关心的agent一个都不在了，就跳过
            continue

        # --- 核心：视角筛选 ---
        visible_cameras = []
        for cam_type in cam_types:
            cam_token = sample['data'][cam_type]
            cam_data = nusc.get('sample_data', cam_token)
            ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
            cam_pose = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_intrinsic = np.array(cam_pose['camera_intrinsic'])

            is_any_agent_visible = False
            for ann in target_anns:
                box = nusc.get_box(ann['token'])
                # 检查是否能成功投影
                if project_box_to_image(box, cam_intrinsic, ego_pose, cam_pose) is not None:
                    is_any_agent_visible = True
                    break  # 只要有一个agent可见，这个视角就有效，无需再检查

            if is_any_agent_visible:
                visible_cameras.append(cam_type)

        # 如果这一帧没有任何一个摄像头能看到agent，就跳过
        if not visible_cameras:
            continue

        # --- 动态生成并保存有效视角的图像 ---
        frame_key = f"frame_{relative_frame_idx:03d}"
        event_manifest["frames"][frame_key] = []

        for cam_type in visible_cameras:
            # 为每个有效视角单独生成一张图
            fig = plt.figure(figsize=(16, 9), dpi=100)
            ax = fig.add_subplot(111)

            cam_token = sample['data'][cam_type]
            nusc.render_sample_data(cam_token, with_anns=False, ax=ax)

            # 重新获取位姿和内参用于绘制
            cam_data = nusc.get('sample_data', cam_token)
            ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
            cam_pose = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_intrinsic = np.array(cam_pose['camera_intrinsic'])

            # 绘制包围盒
            if primary_ann:
                box = nusc.get_box(primary_ann['token'])
                points_2d = project_box_to_image(box, cam_intrinsic, ego_pose, cam_pose)
                if points_2d is not None: draw_projected_box(ax, points_2d, PRIMARY_AGENT_COLOR, 3, label="KEY")

            for token in interacting_agent_tokens:
                ann = get_annotation_for_instance(nusc, sample, token)
                if ann:
                    box = nusc.get_box(ann['token'])
                    points_2d = project_box_to_image(box, cam_intrinsic, ego_pose, cam_pose)
                    if points_2d is not None: draw_projected_box(ax, points_2d, INTERACTING_AGENT_COLOR, 2, label=token[:4])

            ax.set_title(f"{event_id}\n{frame_key} - {cam_type}")
            ax.set_axis_off()

            # --- 保存独立的图片文件 ---
            image_filename = f"{frame_key}_{cam_type}.png"
            output_path = os.path.join(event_output_dir, image_filename)
            plt.savefig(output_path)
            plt.close(fig)

            # 更新manifest
            event_manifest["frames"][frame_key].append(image_filename)

    # 5. 保存该事件的 manifest.json 文件
    with open(os.path.join(event_output_dir, 'manifest.json'), 'w') as f:
        json.dump(event_manifest, f, indent=4)

    return event_output_dir, len(samples_in_event)


if __name__ == '__main__':
    print("Initializing NuScenes SDK...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    print("SDK initialized.")

    print(f"Loading events from '{EVENTS_JSON_PATH}'...")
    with open(EVENTS_JSON_PATH, 'r') as f:
        all_events = json.load(f)
    print(f"Found a total of {len(all_events)} events to visualize.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Optimized visualization results will be saved in '{OUTPUT_DIR}/'")

    for event in tqdm(all_events, desc="Visualizing Events"):
        try:
            start_time = time.time()
            saved_dir, _ = visualize_event_with_filtering(nusc, event, OUTPUT_DIR)
            duration = time.time() - start_time
            if saved_dir:
                event_id_for_log = saved_dir.split('/')[-1]
                tqdm.write(
                    f"Event {event_id_for_log} processed in {duration:.2f}s. Saved to: {saved_dir}")
        except Exception as e:
            event_id_for_log = event.get('scene_token', 'N/A')
            tqdm.write(f"\n[ERROR] Failed to process event {event_id_for_log}: {e}\n{traceback.format_exc()}")
            continue

    print("\nOptimized visualization complete!")
