import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

NUSCENES_DATAROOT = '/data0/senzeyu2/dataset/nuscenes' # nuScenes数据集的根目录
NUSCENES_VERSION = 'v1.0-trainval'            # 使用的数据集版本 ('v1.0-mini' 或 'v1.0-trainval')
EVENTS_JSON_PATH = 'result.json' # 你生成的事件JSON文件
OUTPUT_DIR = '/data0/senzeyu2/dataset/nuscenes/events/'         # 保存可视化结果的文件夹

PRIMARY_AGENT_COLOR = (1, 0, 0) # 红色
INTERACTING_AGENT_COLOR = (0, 0, 1) # 蓝色


def find_closest_sample(nusc, scene_token, target_timestamp):
    """在场景中找到最接近目标时间戳的样本(sample)"""
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
        # 优化：如果时间差开始变大，可以提前退出
        if time_diff > min_time_diff + 0.1:
            break

    return nusc.get('sample', closest_sample_token)


def get_annotation_for_instance(nusc, sample, instance_token):
    """在给定的样本中查找特定实例(instance)的标注(annotation)"""
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token:
            return ann
    return None


def visualize_event(nusc, event_data, output_dir):
    """
    可视化单个事件：拼接6个摄像头视图，并用不同颜色标注 involved agents。
    """
    # 1. 解析事件信息
    event_id = event_data['event_id']
    scene_token = event_id.split('_')[0]
    event_timestamp = event_data['timestamp_start']

    primary_agent_instance = event_data['primary_agent']['agent_id']
    # 只取交互分数最高的那个作为主要交互对象，避免画面混乱
    # 如果想标注所有，可以遍历这个列表
    interacting_agents_instances = [
        agent['agent_id'] for agent in event_data.get('candidate_interacting_agents', [])[:2]  # 最多标注2个
    ]

    # 2. 找到与事件时间最匹配的样本
    sample = find_closest_sample(nusc, scene_token, event_timestamp)

    # 3. 找到涉及的Agent在当前样本中的具体标注
    primary_ann = get_annotation_for_instance(nusc, sample, primary_agent_instance)
    interacting_anns = [
        get_annotation_for_instance(nusc, sample, inst) for inst in interacting_agents_instances
    ]
    interacting_anns = [ann for ann in interacting_anns if ann is not None]  # 过滤掉未找到的

    if not primary_ann:
        print(f"Warning: Could not find primary agent annotation for event {event_id} at timestamp {event_timestamp}")
        return

    # 4. 准备绘图
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), dpi=100)
    axes = axes.ravel()
    cam_types = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
    ]

    for i, cam_type in enumerate(cam_types):
        ax = axes[i]
        cam_token = sample['data'][cam_type]

        # 渲染摄像头图像作为背景 (不带任何标注)
        nusc.render_sample_data(cam_token, with_anns=False, ax=ax)

        # 核心：手动渲染指定颜色的包围盒
        # Primary Agent (红色)
        nusc.render_annotation(primary_ann['token'], ax=ax, box_vis_level=3, color=PRIMARY_AGENT_COLOR, linewidth=3)

        # Interacting Agents (蓝色)
        for ann in interacting_anns:
            nusc.render_annotation(ann['token'], ax=ax, box_vis_level=3, color=INTERACTING_AGENT_COLOR, linewidth=2)

        ax.set_title(cam_type.replace('_', ' '))
        ax.set_axis_off()

    # 5. 保存图像
    fig.suptitle(f'Event: {event_id}\n(Primary: Red, Interacting: Blue)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局为标题留出空间

    output_path = os.path.join(output_dir, f"{event_id}.png")
    plt.savefig(output_path)
    plt.close(fig)  # **非常重要**：在循环中关闭图像，防止内存泄漏


if __name__ == '__main__':
    # 1. 初始化 nuScenes SDK
    print("Initializing NuScenes SDK...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    print("SDK initialized.")

    # 2. 加载事件文件
    print(f"Loading events from '{EVENTS_JSON_PATH}'...")
    if not os.path.exists(EVENTS_JSON_PATH):
        print(f"Error: Events file not found at '{EVENTS_JSON_PATH}'")
        exit()
    with open(EVENTS_JSON_PATH, 'r') as f:
        all_events_by_scene = json.load(f)

    # 3. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Visualization results will be saved in '{OUTPUT_DIR}/'")

    # 4. 遍历所有事件并进行可视化
    # 将所有事件拉平成一个列表，方便使用tqdm显示总进度
    all_events_flat = []
    for scene_name, events_in_scene in all_events_by_scene.items():
        all_events_flat.extend(events_in_scene)

    print(f"Found a total of {len(all_events_flat)} events to visualize.")

    for event in tqdm(all_events_flat, desc="Visualizing Events"):
        try:
            visualize_event(nusc, event, OUTPUT_DIR)
        except Exception as e:
            print(f"\nError processing event {event.get('event_id', 'N/A')}: {e}")
            # 你可以选择在这里记录错误日志
            continue

    print("\nVisualization complete!")
