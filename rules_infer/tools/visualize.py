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


# --- [新] 核心投影函数 ---
def project_box_to_image(box, camera_intrinsic, ego_pose, cam_pose):
    """
    手动将一个3D Box对象投影到图像平面上。
    :param box: nuscenes.utils.data_classes.Box 对象。
    :param camera_intrinsic: 3x3 相机内参矩阵。
    :param ego_pose: 自车位姿记录 (from nusc.get('ego_pose', ...))。
    :param cam_pose: 相机位姿记录 (from nusc.get('calibrated_sensor', ...))。
    :return: (投影后的2D点, 深度值) 或 (None, None) 如果盒子在相机后面。
    """
    # 1. 获取Box在世界坐标系下的8个角点
    corners_3d = box.corners()  # Shape: (3, 8)

    # 2. 从世界坐标系转换到自车坐标系
    ego_pos = np.array(ego_pose['translation'])
    ego_rot = Quaternion(ego_pose['rotation'])
    corners_3d = corners_3d - ego_pos[:, np.newaxis]
    corners_3d = np.dot(ego_rot.inverse.rotation_matrix, corners_3d)

    # 3. 从自车坐标系转换到相机坐标系
    cam_pos = np.array(cam_pose['translation'])
    cam_rot = Quaternion(cam_pose['rotation'])
    corners_3d = corners_3d - cam_pos[:, np.newaxis]
    corners_3d = np.dot(cam_rot.inverse.rotation_matrix, corners_3d)

    # 4. 过滤掉所有在相机后方的角点
    depth = corners_3d[2, :]
    if np.all(depth < 0.1):  # 如果所有点都在相机后面或太近
        return None, None

    # 5. 使用相机内参投影到图像平面
    points_2d = np.dot(camera_intrinsic, corners_3d)

    # 6. 归一化：将 (u, v, d) 转换为 (u/d, v/d)
    # 保证深度大于一个很小的值以避免除零错误
    depth = points_2d[2, :]
    points_2d = points_2d[:2, :] / depth

    return points_2d.T, depth  # Shape: (8, 2) and (8,)


# --- [新] 核心绘制函数 ---
def draw_projected_box(ax, points_2d, color, linewidth):
    """
    在matplotlib的ax上绘制投影后的2D包围盒。
    :param ax: matplotlib.axes.Axes 对象。
    :param points_2d: 8个投影后的2D角点 (shape: 8, 2)。
    :param color: 线的颜色。
    :param linewidth: 线宽。
    """
    # 定义立方体的12条边，连接8个角点
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 侧边
    ]

    for edge in edges:
        start_point = points_2d[edge[0]]
        end_point = points_2d[edge[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                color=color, linewidth=linewidth)


# --- 辅助函数 (保持不变) ---
def find_closest_sample(nusc, scene_token, target_timestamp):
    scene = nusc.get('scene', scene_token)
    current_sample_token = scene['first_sample_token']
    min_time_diff = float('inf')
    closest_sample_token = current_sample_token
    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)
        time_diff = abs(sample['timestamp'] / 1e6 - target_timestamp)
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_sample_token = current_sample_token
        current_sample_token = sample['next']
        if time_diff > min_time_diff + 0.1: break
    return nusc.get('sample', closest_sample_token)


def get_annotation_for_instance(nusc, sample, instance_token):
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token: return ann
    return None


# --- 修改 visualize_event 函数以使用新的投影和绘制逻辑 ---
def visualize_event(nusc, event_data, output_dir):
    event_id = event_data['event_id']
    scene_token = event_id.split('_')[0]
    event_timestamp = event_data['timestamp_start']

    primary_agent_instance = event_data['primary_agent']['agent_id']
    interacting_agents_instances = [agent['agent_id'] for agent in
                                    event_data.get('candidate_interacting_agents', [])[:2]]

    sample = find_closest_sample(nusc, scene_token, event_timestamp)
    primary_ann = get_annotation_for_instance(nusc, sample, primary_agent_instance)
    interacting_anns = [get_annotation_for_instance(nusc, sample, inst) for inst in interacting_agents_instances]
    interacting_anns = [ann for ann in interacting_anns if ann is not None]

    if not primary_ann:
        raise ValueError(f"Could not find primary agent annotation for event {event_id}")

    fig, axes = plt.subplots(2, 3, figsize=(24, 12), dpi=100)
    axes = axes.ravel()
    cam_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']

    for i, cam_type in enumerate(cam_types):
        ax = axes[i]
        cam_token = sample['data'][cam_type]

        # 1. 渲染背景图像 (不变)
        nusc.render_sample_data(cam_token, with_anns=False, ax=ax)

        # 2. 获取该相机和自车的所有位姿和内参信息
        cam_data = nusc.get('sample_data', cam_token)
        ego_pose_data = nusc.get('ego_pose', cam_data['ego_pose_token'])
        cam_pose_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(cam_pose_data['camera_intrinsic'])

        # 3. 绘制 Primary Agent
        primary_box = nusc.get_box(primary_ann['token'])
        points_2d, _ = project_box_to_image(primary_box, camera_intrinsic, ego_pose_data, cam_pose_data)
        if points_2d is not None:
            draw_projected_box(ax, points_2d, PRIMARY_AGENT_COLOR, linewidth=3)

        # 4. 绘制 Interacting Agents
        for ann in interacting_anns:
            box = nusc.get_box(ann['token'])
            points_2d, _ = project_box_to_image(box, camera_intrinsic, ego_pose_data, cam_pose_data)
            if points_2d is not None:
                draw_projected_box(ax, points_2d, INTERACTING_AGENT_COLOR, linewidth=2)

        ax.set_title(cam_type.replace('_', ' '))
        ax.set_axis_off()

    fig.suptitle(f'Event: {event_id}\n(Primary: Red, Interacting: Blue)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, f"{event_id}.png")
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    print("Initializing NuScenes SDK...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    print("SDK initialized.")

    print(f"Loading events from '{EVENTS_JSON_PATH}'...")
    if not os.path.exists(EVENTS_JSON_PATH):
        print(f"Error: Events file not found at '{EVENTS_JSON_PATH}'")
        exit()
    with open(EVENTS_JSON_PATH, 'r') as f:
        all_events_by_scene = json.load(f)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Visualization results will be saved in '{OUTPUT_DIR}/'")

    all_events_flat = []
    for scene_name, events_in_scene in all_events_by_scene.items():
        all_events_flat.extend(events_in_scene)
    print(f"Found a total of {len(all_events_flat)} events to visualize.")

    events_processed = 0
    events_saved = 0
    for event in tqdm(all_events_flat, desc="Visualizing Events"):
        try:
            event_start_time = time.time()
            saved_path = visualize_event(nusc, event, OUTPUT_DIR)
            event_duration = time.time() - event_start_time
            events_processed += 1
            if saved_path:
                events_saved += 1
                tqdm.write(f"Event {event['event_id']} processed and saved in {event_duration:.2f}s.")
        except Exception as e:
            tqdm.write(f"\n[ERROR] Failed to process event {event.get('event_id', 'N/A')}: {e}")
            events_processed += 1
            continue

    print("\nVisualization complete!")
    print(f"Total events attempted: {events_processed}")
    print(f"Total events successfully saved: {events_saved}")

