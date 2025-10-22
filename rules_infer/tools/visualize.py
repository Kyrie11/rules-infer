import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# --- 配置 ---
CONFIG = {
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <--- !!! 确保这里的路径正确 !!!
    'version': 'v1.0-trainval',  # <--- !!! 确保版本与生成json时一致 !!!
    'critical_event_index_file': 'critical_events.json',
    'output_image_dir': 'event_images',  # 保存拼接图片的根文件夹
}

# --- 颜色定义 (BGR格式，因为OpenCV使用BGR) ---
KEY_AGENT_COLOR = (0, 0, 255)  # 红色
INTERACTING_AGENT_COLOR = (255, 0, 0)  # 蓝色
BOX_THICKNESS = 2


def get_sample_token_map_for_scene(nusc, scene_token):
    """
    为给定场景创建一个从帧索引到sample_token的映射，以提高效率。
    """
    scene = nusc.get('scene', scene_token)
    sample_token_map = {}
    current_token = scene['first_sample_token']
    frame_idx = 0
    while current_token:
        sample_token_map[frame_idx] = current_token
        sample = nusc.get('sample', current_token)
        current_token = sample['next']
        frame_idx += 1
    return sample_token_map


def main():
    # --- 1. 初始化 ---
    print("Initializing NuScenes...")
    nusc = NuScenes(version=CONFIG['version'], dataroot=CONFIG['dataroot'], verbose=False)

    print(f"Loading critical events from: {CONFIG['critical_event_index_file']}")
    with open(CONFIG['critical_event_index_file'], 'r') as f:
        critical_events = json.load(f)

    output_dir = CONFIG['output_image_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to subdirectories inside: {output_dir}")

    # --- 2. 遍历所有场景和事件 ---
    total_events = sum(len(events) for events in critical_events.values())
    event_pbar = tqdm(total=total_events, desc="Processing Events")

    for scene_token, events in critical_events.items():
        # 为当前场景预先计算好 帧->token 的映射
        sample_token_map = get_sample_token_map_for_scene(nusc, scene_token)
        scene_record = nusc.get('scene', scene_token)

        for event_idx, event in enumerate(events):
            key_agent_token = event['instance_token']
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            interactions = event['interactions']

            # --- 3. 为当前事件创建一个专属的输出文件夹 ---
            event_folder_name = f"scene_{scene_record['name']}_event_{event_idx}_agent_{key_agent_token[:6]}"
            event_output_path = os.path.join(output_dir, event_folder_name)
            os.makedirs(event_output_path, exist_ok=True)

            # --- 4. 逐帧渲染并保存图片 ---
            for frame_idx in range(start_frame, end_frame):
                sample_token = sample_token_map.get(frame_idx)
                if not sample_token:
                    continue  # 如果帧超出范围，则跳过

                sample = nusc.get('sample', sample_token)

                # 获取当前帧需要关注的交互agent
                interacting_tokens = interactions.get(str(frame_idx), [])

                # 定义摄像头顺序
                cam_names = [
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ]

                rendered_cams = {}

                for cam_name in cam_names:
                    cam_token = sample['data'][cam_name]
                    img_path = nusc.get_sample_data_path(cam_token)
                    img = cv2.imread(img_path)

                    # 获取相机内参，用于投影
                    sd_record = nusc.get('sample_data', cam_token)
                    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                    camera_intrinsic = np.array(cs_record['camera_intrinsic'])

                    # 绘制交互 agent (蓝色)
                    for agent_token in interacting_tokens:
                        ann_token = next((ann for ann in sample['anns'] if
                                          nusc.get('sample_annotation', ann)['instance_token'] == agent_token), None)
                        if ann_token:
                            nusc.render_annotation_in_image(ann_token, img, view_camera_intrinsic=camera_intrinsic,
                                                            box_color=INTERACTING_AGENT_COLOR, thickness=BOX_THICKNESS)

                    # 绘制关键 agent (红色)，确保它被画在最上层
                    key_ann_token = next((ann for ann in sample['anns'] if
                                          nusc.get('sample_annotation', ann)['instance_token'] == key_agent_token),
                                         None)
                    if key_ann_token:
                        nusc.render_annotation_in_image(key_ann_token, img, view_camera_intrinsic=camera_intrinsic,
                                                        box_color=KEY_AGENT_COLOR, thickness=BOX_THICKNESS)

                    rendered_cams[cam_name] = img

                # --- 5. 拼接6个视图 ---
                top_row = np.hstack(
                    [rendered_cams['CAM_FRONT_LEFT'], rendered_cams['CAM_FRONT'], rendered_cams['CAM_FRONT_RIGHT']])
                bottom_row = np.hstack(
                    [rendered_cams['CAM_BACK_LEFT'], rendered_cams['CAM_BACK'], rendered_cams['CAM_BACK_RIGHT']])
                stitched_image = np.vstack([top_row, bottom_row])

                # --- 6. 保存拼接后的图片 ---
                image_filename = f"frame_{frame_idx:04d}.jpg"  # 使用4位补零，方便排序
                image_path = os.path.join(event_output_path, image_filename)
                cv2.imwrite(image_path, stitched_image)

            # --- 7. 完成一个事件，更新进度条 ---
            event_pbar.set_description(f"Saved images to: {event_folder_name}")
            event_pbar.update(1)

    event_pbar.close()
    print("\nVisualization finished!")


# 为了方便调用，我们将 nuscenes-devkit 中的一个渲染函数复制并简化在这里
# 这样就不需要 monkey-patching，代码更独立
def render_annotation_in_image(nusc: NuScenes,
                               annotation_token: str,
                               image: np.ndarray,
                               view_camera_intrinsic: np.ndarray,
                               box_color=(0, 0, 255),
                               thickness=2):
    """
    一个辅助函数，将单个nuScenes标注渲染到OpenCV图像上。
    """
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    ann_record = nusc.get('sample_annotation', annotation_token)
    sample_record = nusc.get('sample', ann_record['sample_token'])

    # 找到此标注对应的相机数据
    # 我们需要从全局坐标转换到当前相机的坐标
    cam_token = None
    for cam_name, token in sample_record['data'].items():
        if 'CAM' in cam_name:
            sd_record = nusc.get('sample_data', token)
            # 检查相机内参是否匹配
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            if np.allclose(cs_record['camera_intrinsic'], view_camera_intrinsic):
                cam_token = token
                break
    if cam_token is None:
        # print("Warning: Could not find matching camera for the given intrinsic matrix.")
        return  # 如果找不到匹配的相机，则无法渲染

    box = nusc.get_box(annotation_token)

    # 将 Box 移动到 ego vehicle 坐标系
    sd_record = nusc.get('sample_data', cam_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(pose_record['rotation']).inverse)

    # 将 Box 移动到相机坐标系
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    box.translate(-np.array(cs_record['translation']))
    box.rotate(Quaternion(cs_record['rotation']).inverse)

    # 检查 Box 是否在相机前方
    if box.center[2] < 0:
        return

    # 投影到2D图像平面并绘制
    box.render_cv2(image, view=view_camera_intrinsic, normalize=True, colors=(box_color, box_color, box_color),
                   linewidth=thickness)


# Monkey-patch NuScenes class to add our method for easier calling.
NuScenes.add_method(render_annotation_in_image)

if __name__ == '__main__':
    main()

