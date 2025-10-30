# visualize_events.py (Adapted for the new social_events.json format)

import os
import json
import time
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用Agg后端，防止在无显示器的服务器上报错
import matplotlib.pyplot as plt
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# --- 配置区 ---
# ### MODIFIED ###: 更新路径以匹配您的环境和新生成的文件
NUSCENES_DATAROOT = '/data0/senzeyu2/dataset/nuscenes'
NUSCENES_VERSION = 'v1.0-trainval'
EVENTS_JSON_PATH = 'social_events.json'  # <-- 指向新生成的JSON文件
OUTPUT_DIR = '/data0/senzeyu2/dataset/nuscenes/events_visualized'  # 使用一个新目录以避免混淆
PRIMARY_AGENT_COLOR = (1, 0, 0)  # Red
INTERACTING_AGENT_COLOR = (0, 0, 1)  # Blue


# --- [稳定] 核心投影与绘制函数 (无需修改) ---
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
    if np.any(depth < 0.1):  # 修正：检查是否有任何一个角点在相机后面或太近
        return None, None
    points_2d = np.dot(camera_intrinsic, corners_3d)
    points_2d[:2, :] = points_2d[:2, :] / points_2d[2, :]
    return points_2d.T, depth


def draw_projected_box(ax, points_2d, color, linewidth):
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for edge in edges:
        start_point, end_point = points_2d[edge[0]], points_2d[edge[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color, linewidth=linewidth)


# --- [稳定] 辅助函数 (无需修改) ---
def get_annotation_for_instance(nusc, sample, instance_token):
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token:
            return ann
    return None


# --- ### NEW ###: 基于帧索引获取样本的函数 ---
def get_sample_tokens_by_frame_range(nusc, scene, start_frame, end_frame):
    """根据帧的起始和结束索引，获取一个场景中的所有sample tokens。"""
    sample_tokens = []
    current_token = scene['first_sample_token']

    # 1. 移动到起始帧
    for _ in range(start_frame):
        sample = nusc.get('sample', current_token)
        current_token = sample['next']
        if not current_token:  # 如果起始帧超出现实，直接返回空
            return []

    # 2. 从起始帧开始收集，直到结束帧
    for i in range(start_frame, end_frame + 1):
        if not current_token:
            break
        sample_tokens.append(current_token)
        sample = nusc.get('sample', current_token)
        current_token = sample['next']

    return sample_tokens


# --- ### CORE REWRITE ###: 可视化整个事件过程的主函数 ---
def visualize_full_event(nusc, event_data, base_output_dir):
    """
    为单个事件创建一个专属目录，并将其持续时间内的每一帧都可视化并保存。
    此函数现在读取新格式的JSON。
    """
    # 1. 从新的event_data格式中提取信息
    scene_token = event_data['scene_token']
    key_agent_token = event_data['key_agent_token']
    interacting_agent_tokens = event_data['interacting_agent_tokens']
    start_frame = event_data['event_start_frame']
    end_frame = event_data['event_end_frame']

    # 创建一个唯一的事件ID用于命名目录
    event_id = f"{scene_token}_{key_agent_token}_{start_frame}-{end_frame}"

    event_output_dir = os.path.join(base_output_dir, event_id)
    os.makedirs(event_output_dir, exist_ok=True)

    # 2. 获取事件持续时间内的所有 nuScenes 'sample'
    scene_record = nusc.get('scene', scene_token)
    samples_in_event = get_sample_tokens_by_frame_range(nusc, scene_record, start_frame, end_frame)

    if not samples_in_event:
        tqdm.write(f"  [Warning] No samples found for event {event_id}. Skipping.")
        return None, 0

    # 3. 遍历事件中的每一帧 (sample)，生成并保存图像
    for frame_idx, sample_token in enumerate(samples_in_event):
        sample = nusc.get('sample', sample_token)

        # 为当前帧寻找Agent的标注
        primary_ann = get_annotation_for_instance(nusc, sample, key_agent_token)
        interacting_anns = [get_annotation_for_instance(nusc, sample, inst) for inst in interacting_agent_tokens]
        interacting_anns = [ann for ann in interacting_anns if ann is not None]

        fig, axes = plt.subplots(2, 3, figsize=(24, 12), dpi=100)
        axes = axes.ravel()
        cam_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']

        for i, cam_type in enumerate(cam_types):
            ax = axes[i]
            cam_token = sample['data'][cam_type]
            nusc.render_sample_data(cam_token, with_anns=False, ax=ax)

            cam_data = nusc.get('sample_data', cam_token)
            ego_pose_data = nusc.get('ego_pose', cam_data['ego_pose_token'])
            cam_pose_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            camera_intrinsic = np.array(cam_pose_data['camera_intrinsic'])

            if primary_ann:
                primary_box = nusc.get_box(primary_ann['token'])
                points_2d, _ = project_box_to_image(primary_box, camera_intrinsic, ego_pose_data, cam_pose_data)
                if points_2d is not None:
                    draw_projected_box(ax, points_2d, PRIMARY_AGENT_COLOR, linewidth=3)

            for ann in interacting_anns:
                box = nusc.get_box(ann['token'])
                points_2d, _ = project_box_to_image(box, camera_intrinsic, ego_pose_data, cam_pose_data)
                if points_2d is not None:
                    draw_projected_box(ax, points_2d, INTERACTING_AGENT_COLOR, linewidth=2)

            ax.set_title(cam_type.replace('_', ' '))
            ax.set_axis_off()

        scene_name = event_data.get('scene_name', 'N/A')
        fig.suptitle(f'Scene: {scene_name}\nEvent ID: {event_id}\nFrame: {frame_idx + 1}/{len(samples_in_event)}',
                     fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(event_output_dir, f"frame_{frame_idx:03d}.png")
        plt.savefig(output_path)
        plt.close(fig)

    return event_output_dir, len(samples_in_event)


if __name__ == '__main__':
    print("Initializing NuScenes SDK...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    print("SDK initialized.")

    print(f"Loading events from '{EVENTS_JSON_PATH}'...")
    with open(EVENTS_JSON_PATH, 'r') as f:
        # ### MODIFIED ###: 新的JSON是扁平列表，直接加载即可
        all_events = json.load(f)
    print(f"Found a total of {len(all_events)} events to visualize.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Visualization results will be saved in '{OUTPUT_DIR}/'")

    events_processed = 0
    # ### MODIFIED ###: 直接遍历加载的列表
    for event in tqdm(all_events, desc="Visualizing Events"):
        try:
            event_start_time = time.time()
            saved_dir, num_frames = visualize_full_event(nusc, event, OUTPUT_DIR)
            event_duration = time.time() - event_start_time
            events_processed += 1
            if saved_dir:
                event_id_for_log = f"{event['scene_token']}_{event['key_agent_token']}_{event['event_start_frame']}"
                tqdm.write(
                    f"Event {event_id_for_log} processed ({num_frames} frames) in {event_duration:.2f}s. Saved to: {saved_dir}")

        except Exception as e:
            event_id_for_log = event.get('scene_token', 'N/A') + '_' + event.get('key_agent_token', 'N/A')
            tqdm.write(f"\n[ERROR] Failed to process event {event_id_for_log}: {e}")
            import traceback

            tqdm.write(traceback.format_exc())
            continue

    print("\nVisualization complete!")
