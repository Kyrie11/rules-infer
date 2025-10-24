import matplotlib

matplotlib.use('Agg')  # 设置非交互式后端

import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple
import numpy as np  # <-- 新增导入

# 确保 nuscenes-devkit 已安装
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# --- 配置 (保持不变) ---
CONFIG = {
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',
    'version': 'v1.0-trainval',
    'json_path': 'critical_events.json',
    'output_dir': 'critical_event_visuals'
}

# --- 常量 (保持不变) ---
COLOR_MAIN_AGENT = 'red'
COLOR_INTERACTING_AGENT = 'deepskyblue'
CAMERAS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]


def get_sample_tokens_in_scene(nusc: NuScenes, scene_token: str) -> List[str]:
    # ... 此函数保持不变 ...
    scene = nusc.get('scene', scene_token)
    sample_tokens = []
    current_token = scene['first_sample_token']
    while current_token:
        sample_tokens.append(current_token)
        sample = nusc.get('sample', current_token)
        current_token = sample['next']
    return sample_tokens


# --- 新增的辅助函数 ---
def draw_box_on_ax(nusc: NuScenes, ann_token: str, cam_token: str, ax: plt.Axes, color: str):
    """
    将一个指定的标注框 (annotation) 绘制到指定的摄像头图像的子图 (Axes) 上。

    :param nusc: NuScenes API 对象。
    :param ann_token: 要绘制的 sample_annotation 的 token。
    :param cam_token: 目标摄像头的 sample_data 的 token。
    :param ax: matplotlib 的 Axes 对象。
    :param color: 框的颜色。
    """
    # 1. 获取标注框对象
    ann_record = nusc.get('sample_annotation', ann_token)
    box = nusc.get_box(ann_token)

    # 2. 获取摄像头标定和位姿数据
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])

    # 3. 将 Box 从全局坐标系转换到摄像头坐标系
    # 3.1 从全局坐标系转换到自车坐标系
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(pose_record['rotation']).inverse)
    # 3.2 从自车坐标系转换到摄像头坐标系
    box.translate(-np.array(cs_record['translation']))
    box.rotate(Quaternion(cs_record['rotation']).inverse)

    # 4. 检查框是否在相机前方 (z > 0)
    # 如果框的所有角点都在相机后面，就不绘制它
    if np.all(box.corners()[2, :] < 0):
        return

    # 5. 使用 Box 自带的 render 方法绘制到指定的 ax 上
    # view 参数是相机的内参矩阵
    # normalize=True 会处理投影所需的除以z的操作
    # colors 参数是一个元组 (front, back, sides)
    box.render(ax, view=np.array(cs_record['camera_intrinsic']), normalize=True,
               colors=(color, color, color), linewidth=2)


# --- 修改后的主函数 ---
def visualize_events(config: dict):
    """
    主函数，读取JSON文件并为每个事件生成可视化视频帧。
    """
    print("Initializing NuScenes...")
    nusc = NuScenes(version=config['version'], dataroot=config['dataroot'], verbose=False)

    print(f"Loading critical events from {config['json_path']}...")
    try:
        with open(config['json_path'], 'r') as f:
            critical_events = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {config['json_path']}")
        return

    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"Output will be saved to {config['output_dir']}")

    scene_progress = tqdm(critical_events.items(), desc="Processing Scenes")

    for scene_token, events in scene_progress:
        scene_progress.set_postfix_str(f"Scene: {nusc.get('scene', scene_token)['name']}")

        sample_tokens_in_scene = get_sample_tokens_in_scene(nusc, scene_token)

        for event_idx, event in enumerate(events):
            main_agent_token = event['instance_token']
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            interactions = event['interactions']

            event_folder_name = f"scene_{nusc.get('scene', scene_token)['name']}_event_{event_idx}_agent_{main_agent_token[:6]}"
            event_output_dir = os.path.join(config['output_dir'], event_folder_name)
            os.makedirs(event_output_dir, exist_ok=True)

            print(
                f"\nProcessing Event {event_idx + 1}/{len(events)} in Scene {nusc.get('scene', scene_token)['name']}...")

            frame_progress = tqdm(range(start_frame, end_frame), desc="  - Rendering Frames", leave=False)
            for frame_idx in frame_progress:
                if frame_idx >= len(sample_tokens_in_scene):
                    continue

                sample_token = sample_tokens_in_scene[frame_idx]
                sample = nusc.get('sample', sample_token)
                interacting_tokens = interactions.get(str(frame_idx), [])

                fig, axes = plt.subplots(2, 3, figsize=(18, 8))
                axes = axes.ravel()

                for i, cam_name in enumerate(CAMERAS):
                    ax = axes[i]
                    cam_token = sample['data'][cam_name]

                    nusc.render_sample_data(cam_token, with_anns=False, ax=ax)
                    ax.set_title(cam_name)
                    ax.axis('off')

                    # --- 绘制主角的框 (使用新方法) ---
                    main_agent_ann_token = None
                    for ann_token in sample['anns']:
                        ann = nusc.get('sample_annotation', ann_token)
                        if ann['instance_token'] == main_agent_token:
                            main_agent_ann_token = ann_token
                            break

                    if main_agent_ann_token:
                        draw_box_on_ax(nusc, main_agent_ann_token, cam_token, ax, COLOR_MAIN_AGENT)

                    # --- 绘制交互对象的框 (使用新方法) ---
                    for other_token in interacting_tokens:
                        other_ann_token = None
                        for ann_token in sample['anns']:
                            ann = nusc.get('sample_annotation', ann_token)
                            if ann['instance_token'] == other_token:
                                other_ann_token = ann_token
                                break
                        if other_ann_token:
                            draw_box_on_ax(nusc, other_ann_token, cam_token, ax, COLOR_INTERACTING_AGENT)

                plt.tight_layout()
                output_path = os.path.join(event_output_dir, f"frame_{frame_idx:04d}.jpg")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

    print("\nVisualization finished for all events.")


if __name__ == '__main__':
    if not os.path.exists(CONFIG['dataroot']):
        print(f"Error: NuScenes dataroot not found at '{CONFIG['dataroot']}'. Please update the CONFIG.")
    else:
        visualize_events(CONFIG)

