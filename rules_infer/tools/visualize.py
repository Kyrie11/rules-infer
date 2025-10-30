import os
import json
import time
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# --- 配置区 (保持不变) ---
NUSCENES_DATAROOT = '/data0/senzeyu2/dataset/nuscenes'
NUSCENES_VERSION = 'v1.0-trainval'
EVENTS_JSON_PATH = 'result.json'
OUTPUT_DIR = '/data0/senzeyu2/dataset/nuscenes/events'
PRIMARY_AGENT_COLOR = (1, 0, 0)  # Red
INTERACTING_AGENT_COLOR = (0, 0, 1)  # Blue


# --- [稳定] 核心投影与绘制函数 (保持不变) ---
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
    if np.all(depth < 0.1):
        return None, None
    points_2d = np.dot(camera_intrinsic, corners_3d)
    depth = points_2d[2, :]
    points_2d = points_2d[:2, :] / depth
    return points_2d.T, depth


def draw_projected_box(ax, points_2d, color, linewidth):
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for edge in edges:
        start_point, end_point = points_2d[edge[0]], points_2d[edge[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color, linewidth=linewidth)


# --- [稳定] 辅助函数 (保持不变) ---
def get_annotation_for_instance(nusc, sample, instance_token):
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token:
            return ann
    return None


# --- [新] 获取时间段内所有样本的函数 ---
def get_samples_in_timespan(nusc, scene_token, ts_start, ts_end):
    """获取一个场景中，位于指定时间戳范围内的所有sample tokens。"""
    samples_in_range = []
    scene = nusc.get('scene', scene_token)
    current_sample_token = scene['first_sample_token']

    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)
        sample_ts = sample['timestamp'] / 1e6

        if sample_ts >= ts_start and sample_ts <= ts_end:
            samples_in_range.append(current_sample_token)

        # 优化: 如果当前样本时间已经远超结束时间，可以提前退出
        if sample_ts > ts_end + 1.0:
            break

        current_sample_token = sample['next']

    return samples_in_range


# --- [核心重构] 可视化整个事件过程的主函数 ---
def visualize_full_event(nusc, event_data, base_output_dir):
    """
    为单个事件创建一个专属目录，并将其持续时间内的每一帧都可视化并保存。
    """
    event_id = event_data['event_id']
    scene_token = event_id.split('_')[0]
    ts_start = event_data['timestamp_start']
    ts_end = event_data['timestamp_end']

    # 1. 为此事件创建专属的输出目录
    event_output_dir = os.path.join(base_output_dir, event_id)
    os.makedirs(event_output_dir, exist_ok=True)

    # 2. 获取事件涉及的Agent实例
    primary_agent_instance = event_data['primary_agent']['agent_id']
    interacting_agents_instances = [agent['agent_id'] for agent in
                                    event_data.get('candidate_interacting_agents', [])[:2]]

    # 3. 找到事件持续时间内的所有 nuScenes 'sample'
    samples_in_event = get_samples_in_timespan(nusc, scene_token, ts_start, ts_end)

    if not samples_in_event:
        tqdm.write(f"  [Warning] No samples found for event {event_id} in its time range. Skipping.")
        return

    # 4. 遍历事件中的每一帧 (sample)，生成并保存图像
    for frame_idx, sample_token in enumerate(samples_in_event):
        sample = nusc.get('sample', sample_token)

        # 为当前帧寻找Agent的标注
        primary_ann = get_annotation_for_instance(nusc, sample, primary_agent_instance)
        interacting_anns = [get_annotation_for_instance(nusc, sample, inst) for inst in interacting_agents_instances]
        interacting_anns = [ann for ann in interacting_anns if ann is not None]

        fig, axes = plt.subplots(2, 3, figsize=(24, 12), dpi=100)
        axes = axes.ravel()
        cam_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']

        for i, cam_type in enumerate(cam_types):
            ax = axes[i]
            cam_token = sample['data'][cam_type]

            # 渲染背景
            nusc.render_sample_data(cam_token, with_anns=False, ax=ax)

            # 获取位姿和内参
            cam_data = nusc.get('sample_data', cam_token)
            ego_pose_data = nusc.get('ego_pose', cam_data['ego_pose_token'])
            cam_pose_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            camera_intrinsic = np.array(cam_pose_data['camera_intrinsic'])

            # 手动投影和绘制主车
            if primary_ann:
                primary_box = nusc.get_box(primary_ann['token'])
                points_2d, _ = project_box_to_image(primary_box, camera_intrinsic, ego_pose_data, cam_pose_data)
                if points_2d is not None:
                    draw_projected_box(ax, points_2d, PRIMARY_AGENT_COLOR, linewidth=3)

            # 手动投影和绘制交互车辆
            for ann in interacting_anns:
                box = nusc.get_box(ann['token'])
                points_2d, _ = project_box_to_image(box, camera_intrinsic, ego_pose_data, cam_pose_data)
                if points_2d is not None:
                    draw_projected_box(ax, points_2d, INTERACTING_AGENT_COLOR, linewidth=2)

            ax.set_title(cam_type.replace('_', ' '))
            ax.set_axis_off()

        # 添加标题并保存
        fig.suptitle(f'Event: {event_id}\nFrame: {frame_idx:03d}', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(event_output_dir, f"frame_{frame_idx:03d}.png")
        plt.savefig(output_path)
        plt.close(fig)  # **极其重要**：在循环内部关闭图像，防止内存爆炸！

    return event_output_dir, len(samples_in_event)  # 返回路径和帧数，方便主循环打印信息


if __name__ == '__main__':
    print("Initializing NuScenes SDK...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    print("SDK initialized.")

    print(f"Loading events from '{EVENTS_JSON_PATH}'...")
    with open(EVENTS_JSON_PATH, 'r') as f:
        all_events_by_scene = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Visualization results will be saved in '{OUTPUT_DIR}/'")

    all_events_flat = []
    for scene_name, events_in_scene in all_events_by_scene.items():
        all_events_flat.extend(events_in_scene)
    print(f"Found a total of {len(all_events_flat)} events to visualize.")

    events_processed = 0
    for event in tqdm(all_events_flat, desc="Visualizing Events"):
        try:
            event_start_time = time.time()

            # 调用新的主函数
            saved_dir, num_frames = visualize_full_event(nusc, event, OUTPUT_DIR)

            event_duration = time.time() - event_start_time
            events_processed += 1

            if saved_dir:
                tqdm.write(
                    f"Event {event['event_id']} processed ({num_frames} frames) in {event_duration:.2f}s. Saved to: {saved_dir}")

        except Exception as e:
            tqdm.write(f"\n[ERROR] Failed to process event {event.get('event_id', 'N/A')}: {e}")
            import traceback

            tqdm.write(traceback.format_exc())  # 打印详细的错误堆栈
            events_processed += 1
            continue

    print("\nVisualization complete!")
    print(f"Total events attempted: {events_processed}")
