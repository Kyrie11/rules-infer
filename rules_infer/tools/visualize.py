import os
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# 确保 nuscenes-devkit 已安装
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

CONFIG = {
    # --- 数据和文件路径 ---
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <--- !!! 你的 nuscenes 数据集路径 !!!
    'version': 'v1.0-trainval',  # <--- !!! 确保与生成json时版本一致 !!!
    'critical_event_file': 'critical_events.json',  # 你生成的事件索引文件
    'output_dir': '/data0/senzeyu2/dataset/nuscenes/critical',  # 保存视频帧的总文件夹
}

# 定义颜色 (OpenCV使用 BGR 格式)
COLOR_KEY_AGENT = (0, 0, 255)  # 红色
COLOR_INTERACTING_AGENT = (255, 0, 0)  # 蓝色

# 定义摄像头顺序用于拼接
CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]


def get_scene_samples(nusc: NuScenes, scene_token: str):
    """
    获取一个场景中所有样本(sample)并按时间顺序排列。
    """
    scene = nusc.get('scene', scene_token)
    samples = []
    current_sample_token = scene['first_sample_token']
    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)
        samples.append(sample)
        current_sample_token = sample['next']
    return samples


def draw_3d_box(nusc: NuScenes, image: np.ndarray, ann_token: str, color: tuple):
    """
    在单张相机图像上绘制一个3D边界框。
    """
    # 1. 获取标注记录和相机数据
    ann_record = nusc.get('sample_annotation', ann_token)
    sample_data_token = nusc.get('sample', ann_record['sample_token'])['data'][cam_name]
    cs_record = nusc.get('calibrated_sensor', nusc.get('sample_data', sample_data_token)['calibrated_sensor_token'])

    # 2. 加载3D Box
    box = nusc.get_box(ann_token)

    # 3. 将Box渲染到图像上
    # nuscenes-devkit > v1.1.9 版本有更方便的 box.render_cv2 方法
    # 这里使用更通用的方法，手动投影
    box.render_cv2(image, view=np.array(cs_record['camera_intrinsic']), normalize=True, color=color, thickness=2)


def stitch_six_views(cam_images: dict) -> np.ndarray:
    """
    将6个摄像头视图拼接成一个 2x3 的网格图。
    """
    # 获取单张图片的尺寸
    h, w, _ = cam_images['CAM_FRONT'].shape

    # 创建一个空的画布
    grid_image = np.zeros((2 * h, 3 * w, 3), dtype=np.uint8)

    # 拼接图像
    grid_image[0:h, 0:w] = cam_images['CAM_FRONT_LEFT']
    grid_image[0:h, w:2 * w] = cam_images['CAM_FRONT']
    grid_image[0:h, 2 * w:3 * w] = cam_images['CAM_FRONT_RIGHT']
    grid_image[h:2 * h, 0:w] = cam_images['CAM_BACK_LEFT']
    grid_image[h:2 * h, w:2 * w] = cam_images['CAM_BACK']
    grid_image[h:2 * h, 2 * w:3 * w] = cam_images['CAM_BACK_RIGHT']

    return grid_image


def visualize_event(nusc: NuScenes, scene_token: str, event_data: dict, event_idx: int, output_root: Path):
    """
    为单个关键事件生成所有帧的可视化结果。
    """
    # --- 1. 解析事件数据 ---
    key_agent_token = event_data['instance_token']
    start_frame = event_data['start_frame']
    end_frame = event_data['end_frame']
    reason = event_data['reason']
    interactions = event_data['interactions']

    # --- 2. 创建输出目录 ---
    event_folder_name = f"scene_{nusc.get('scene', scene_token)['name']}_evt{event_idx}_{reason}_{key_agent_token[:6]}"
    event_output_dir = output_root / event_folder_name
    event_output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nProcessing event: {event_folder_name}")

    # --- 3. 预加载场景的所有样本信息 ---
    scene_samples = get_scene_samples(nusc, scene_token)

    # --- 4. 逐帧处理和渲染 ---
    for frame_idx in tqdm(range(start_frame, end_frame), desc="Rendering frames"):
        if frame_idx >= len(scene_samples):
            print(f"Warning: frame_idx {frame_idx} is out of bounds for scene {scene_token}. Skipping.")
            continue

        sample = scene_samples[frame_idx]

        # 建立一个从 instance_token 到当前帧 annotation_token 的快速查找映射
        inst_to_ann_map = {}
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            inst_to_ann_map[ann['instance_token']] = ann_token

        # 获取当前帧的交互对象
        interacting_tokens = interactions.get(str(frame_idx), [])

        cam_images = {}
        # 渲染6个摄像头视图
        for cam_name in CAM_ORDER:
            sample_data_token = sample['data'][cam_name]
            image_path = nusc.get_sample_data_path(sample_data_token)
            image = cv2.imread(str(image_path))  # 使用 str() 转换 Path 对象

            # 绘制关键Agent (红色)
            key_ann_token = inst_to_ann_map.get(key_agent_token)
            if key_ann_token:
                nusc.render_annotation_in_image(key_ann_token, image, cam_name, box_color=COLOR_KEY_AGENT)

            # 绘制交互Agent (蓝色)
            for other_token in interacting_tokens:
                other_ann_token = inst_to_ann_map.get(other_token)
                if other_ann_token:
                    nusc.render_annotation_in_image(other_ann_token, image, cam_name, box_color=COLOR_INTERACTING_AGENT)

            cam_images[cam_name] = image

        # 拼接成大图
        stitched_image = stitch_six_views(cam_images)

        # 在大图上添加文字信息
        text_info = f"Scene: {nusc.get('scene', scene_token)['name']} | Frame: {frame_idx} | Event: {reason}"
        cv2.putText(stitched_image, text_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(stitched_image, f"Key Agent (RED): {key_agent_token[:6]}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    COLOR_KEY_AGENT, 2, cv2.LINE_AA)

        # --- 5. 保存当前帧 ---
        output_frame_path = event_output_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(output_frame_path), stitched_image)


def main():
    print("--- Starting Critical Event Visualization ---")

    # 初始化 NuScenes
    print(f"Loading NuScenes dataset from {CONFIG['dataroot']}...")
    nusc = NuScenes(version=CONFIG['version'], dataroot=CONFIG['dataroot'], verbose=False)

    # 加载事件文件
    event_file = Path(CONFIG['critical_event_file'])
    if not event_file.exists():
        print(f"Error: Critical event file not found at {event_file}")
        return
    with open(event_file, 'r') as f:
        critical_events = json.load(f)
    print(f"Loaded {sum(len(v) for v in critical_events.values())} critical events from {event_file}")

    # 创建主输出目录
    output_root = Path(CONFIG['output_dir'])
    output_root.mkdir(exist_ok=True)

    # 遍历所有事件并进行可视化
    for scene_token, events in critical_events.items():
        for i, event in enumerate(events):
            visualize_event(nusc, scene_token, event, i, output_root)

    print("\n--- Visualization process finished! ---")
    print(f"Results saved in: {output_root.resolve()}")
    print("\nTo create a video from the image frames, you can use ffmpeg.")
    print("Example command for one event folder:")
    print("ffmpeg -framerate 5 -i 'path/to/your/event_folder/frame_%04d.png' -c:v libx264 -pix_fmt yuv420p output.mp4")


if __name__ == '__main__':
    main()
