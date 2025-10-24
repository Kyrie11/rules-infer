import matplotlib
matplotlib.use('Agg')  # <-- 在导入 pyplot 之前添加这一行

import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
# ... a lot of other imports

from typing import List, Tuple

# 确保 nuscenes-devkit 已安装
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# --- 配置 ---
CONFIG = {
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <--- !!! 确保这是你的NuScenes数据根目录
    'version': 'v1.0-trainval',  # <--- !!! 确保这与生成JSON时使用的版本一致
    'json_path': 'critical_events.json',  # <--- 输入的JSON文件
    'output_dir': '/data0/senzeyu2/dataset/nuscenes/critical'  # <--- 保存可视化结果的根文件夹
}

# 定义颜色
COLOR_MAIN_AGENT = 'red'
COLOR_INTERACTING_AGENT = 'deepskyblue'  # 使用更亮的蓝色以示区分
CAMERAS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]


def get_sample_tokens_in_scene(nusc: NuScenes, scene_token: str) -> List[str]:
    """
    高效地获取一个场景中所有sample_token的有序列表。
    """
    scene = nusc.get('scene', scene_token)
    sample_tokens = []
    current_token = scene['first_sample_token']
    while current_token:
        sample_tokens.append(current_token)
        sample = nusc.get('sample', current_token)
        current_token = sample['next']
    return sample_tokens


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

    # 创建主输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"Output will be saved to {config['output_dir']}")

    # 使用tqdm来显示场景处理的进度
    scene_progress = tqdm(critical_events.items(), desc="Processing Scenes")

    for scene_token, events in scene_progress:
        scene_progress.set_postfix_str(f"Scene: {nusc.get('scene', scene_token)['name']}")

        # 1. 为当前场景预先获取所有 sample_token，避免在循环中重复查找
        sample_tokens_in_scene = get_sample_tokens_in_scene(nusc, scene_token)

        # 遍历该场景下的所有事件
        for event_idx, event in enumerate(events):
            main_agent_token = event['instance_token']
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            interactions = event['interactions']  # 键是字符串形式的帧号

            # 2. 为每个事件创建独立的文件夹
            event_folder_name = f"scene_{nusc.get('scene', scene_token)['name']}_event_{event_idx}_agent_{main_agent_token[:6]}"
            event_output_dir = os.path.join(config['output_dir'], event_folder_name)
            os.makedirs(event_output_dir, exist_ok=True)

            print(
                f"\nProcessing Event {event_idx + 1}/{len(events)} in Scene {nusc.get('scene', scene_token)['name']}...")
            print(f"  - Main Agent: {main_agent_token}")
            print(f"  - Frames: {start_frame} to {end_frame}")
            print(f"  - Saving to: {event_output_dir}")

            # 3. 遍历事件的每一帧
            frame_progress = tqdm(range(start_frame, end_frame), desc="  - Rendering Frames", leave=False)
            for frame_idx in frame_progress:
                # 检查帧索引是否有效
                if frame_idx >= len(sample_tokens_in_scene):
                    print(
                        f"Warning: Frame index {frame_idx} is out of bounds for scene {scene_token} (max: {len(sample_tokens_in_scene) - 1}). Skipping.")
                    continue

                sample_token = sample_tokens_in_scene[frame_idx]
                sample = nusc.get('sample', sample_token)

                # 获取当前帧的交互对象token列表
                interacting_tokens = interactions.get(str(frame_idx), [])

                # 4. 创建 3x2 的画布
                fig, axes = plt.subplots(2, 3, figsize=(18, 8))
                axes = axes.ravel()  # 将 2x3 数组扁平化为 1D 数组，方便遍历

                # 5. 遍历6个摄像头
                for i, cam_name in enumerate(CAMERAS):
                    ax = axes[i]
                    cam_token = sample['data'][cam_name]

                    # 绘制图像和标注框
                    nusc.render_sample_data(cam_token, with_anns=False, ax=ax)  # 先只画背景图
                    ax.set_title(cam_name)
                    ax.axis('off')  # 关闭坐标轴

                    # --- 绘制主角的框 ---
                    # 找到主角在该帧的 annotation token
                    main_agent_ann_token = None
                    for ann_token in sample['anns']:
                        ann = nusc.get('sample_annotation', ann_token)
                        if ann['instance_token'] == main_agent_token:
                            main_agent_ann_token = ann_token
                            break

                    if main_agent_ann_token:
                        # 使用 nusc.render_annotation 来绘制，它能处理好投影和可见性
                        nusc.render_annotation(main_agent_ann_token, box_vis_level=1, out_path=None, ax=ax,
                                               colors=(COLOR_MAIN_AGENT, 'white', 'black'))

                    # --- 绘制交互对象的框 ---
                    for other_token in interacting_tokens:
                        other_ann_token = None
                        for ann_token in sample['anns']:
                            ann = nusc.get('sample_annotation', ann_token)
                            if ann['instance_token'] == other_token:
                                other_ann_token = ann_token
                                break
                        if other_ann_token:
                            nusc.render_annotation(other_ann_token, box_vis_level=1, out_path=None, ax=ax,
                                                   colors=(COLOR_INTERACTING_AGENT, 'white', 'black'))

                # 6. 调整布局并保存图像
                plt.tight_layout()
                output_path = os.path.join(event_output_dir, f"frame_{frame_idx:04d}.jpg")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)  # **非常重要**：关闭图像以释放内存

    print("\nVisualization finished for all events.")


if __name__ == '__main__':
    # 确保你的配置是正确的
    if not os.path.exists(CONFIG['dataroot']):
        print(f"Error: NuScenes dataroot not found at '{CONFIG['dataroot']}'. Please update the CONFIG.")
    else:
        visualize_events(CONFIG)

