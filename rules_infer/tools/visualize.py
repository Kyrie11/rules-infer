import json
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 确保 nuscenes-devkit 已安装
from nuscenes.nuscenes import NuScenes
from nuscenes.explorer import NuScenesExplorer  ### NEW ###: Import the correct class for visualization

# ----------------------------------
# 1. 配置
# ----------------------------------
CONFIG = {
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <--- !!! 确保这是你的NuScenes数据根目录
    'version': 'v1.0-trainval',  # <--- !!! 确保这与生成JSON时使用的版本一致
    'event_file': 'critical_events.json',  # 输入的JSON事件文件
    'output_dir': 'critical_event_visualizations',  # 保存可视化结果的总文件夹
}

# ----------------------------------
# 2. 辅助函数
# ----------------------------------

# 缓存，避免重复计算，大大提高效率
_scene_to_samples_cache = {}


def get_sample_token_by_frame(nusc: NuScenes, scene_token: str, frame_idx: int) -> str:
    """
    根据场景token和帧序号（从0开始）获取sample_token。
    使用了缓存来加速查找。
    """
    if scene_token not in _scene_to_samples_cache:
        scene = nusc.get('scene', scene_token)
        sample_tokens = []
        current_sample_token = scene['first_sample_token']
        while current_sample_token:
            sample_tokens.append(current_sample_token)
            sample = nusc.get('sample', current_sample_token)
            current_sample_token = sample['next']
        _scene_to_samples_cache[scene_token] = sample_tokens

    scene_samples = _scene_to_samples_cache[scene_token]
    if 0 <= frame_idx < len(scene_samples):
        return scene_samples[frame_idx]
    else:
        raise IndexError(
            f"Frame index {frame_idx} is out of bounds for scene {scene_token} with {len(scene_samples)} frames.")


def get_annotation_for_instance(nusc: NuScenes, sample_token: str, instance_token: str) -> str:
    """
    在给定的sample中查找特定instance的annotation_token。
    """
    sample = nusc.get('sample', sample_token)
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token:
            return ann_token
    return None


# ----------------------------------
# 3. 主可视化函数
# ----------------------------------
def main():
    print("Initializing NuScenes API...")
    nusc = NuScenes(version=CONFIG['version'], dataroot=CONFIG['dataroot'], verbose=False)

    print(f"Loading critical events from '{CONFIG['event_file']}'...")
    try:
        with open(CONFIG['event_file'], 'r') as f:
            critical_events_by_scene = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{CONFIG['event_file']}' was not found.")
        print("Please make sure you have run the event detection script first and the file is in the correct location.")
        return

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"Visualizations will be saved to '{CONFIG['output_dir']}'")

    cam_order = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    total_events = sum(len(events) for events in critical_events_by_scene.values())
    event_pbar = tqdm(total=total_events, desc="Processing Events")

    for scene_token, events in critical_events_by_scene.items():
        scene_record = nusc.get('scene', scene_token)
        scene_name = scene_record['name']

        for i, event in enumerate(events):
            instance_token = event['instance_token']
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            interactions = event['interactions']
            reason = event.get('reason', 'event')

            event_folder_name = f"scene-{scene_name}_agent-{instance_token[:8]}_{reason}_{i}"
            event_output_path = os.path.join(CONFIG['output_dir'], event_folder_name)
            os.makedirs(event_output_path, exist_ok=True)

            event_pbar.set_description(f"Event: {event_folder_name}")

            for frame_idx in range(start_frame, end_frame):
                try:
                    sample_token = get_sample_token_by_frame(nusc, scene_token, frame_idx)
                except IndexError as e:
                    print(f"Warning: Skipping frame. {e}")
                    continue

                interacting_tokens = interactions.get(str(frame_idx), [])

                fig, axes = plt.subplots(2, 3, figsize=(24, 12))
                axes = axes.ravel()

                for ax, cam_name in zip(axes, cam_order):
                    sample = nusc.get('sample', sample_token)
                    cam_data_token = sample['data'][cam_name]

                    # 使用render_sample_data方法直接渲染图像
                    nusc.render_sample_data(cam_data_token, with_anns=False, ax=ax)
                    ax.set_title(cam_name)

                    # 渲染主角的包围框 (红色)
                    key_agent_ann_token = get_annotation_for_instance(nusc, sample_token, instance_token)
                    if key_agent_ann_token:
                        # 使用 render_annotation 方法渲染
                        nusc.render_annotation(key_agent_ann_token, ax=ax, color='red', linewidth=3)

                    # 渲染交互对象的包围框 (蓝色)
                    for inter_token in interacting_tokens:
                        inter_ann_token = get_annotation_for_instance(nusc, sample_token, inter_token)
                        if inter_ann_token:
                            nusc.render_annotation(inter_ann_token, ax=ax, color='blue', linewidth=2)

                fig.suptitle(f"Scene: {scene_name} | Frame: {frame_idx}\nKey Agent (RED): {instance_token[:8]}",
                             fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                output_image_path = os.path.join(event_output_path, f"frame_{frame_idx:04d}.png")
                plt.savefig(output_image_path)
                plt.close(fig)

            event_pbar.update(1)

    event_pbar.close()
    print("\nVisualization process finished successfully.")
    print(f"All event clips have been saved in '{CONFIG['output_dir']}'")


if __name__ == '__main__':
    main()
