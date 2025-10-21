import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion

# --- 配置 ---
CONFIG = {
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <--- !!! 确认路径 !!!
    'version': 'v1.0-trainval',  # <--- !!! 确认版本 !!!
    'critical_event_index_file': 'critical_events.json',
    'visualization_output_dir': '/data0/senzeyu2/dataset/nuscenes/critical_event',  # 新的输出目录

    'history_len': 8,
    'future_len': 12,

    # --- 标注颜色 (B, G, R) ---
    'color_key_agent_box': (0, 0, 255),  # 红色
    'color_interaction_agent_box': (255, 0, 0),  # 蓝色
    'color_gt_trajectory': (0, 0, 255),  # 红色
    'color_pred_trajectory': (0, 255, 0),  # 绿色

    # --- 网格布局参数 ---
    'grid_scale': 0.5,  # 将每个摄像头图像缩小的比例
    'grid_layout': [
        ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    ]
}


# (这里可以复用之前脚本中的 project_trajectory_to_image 函数)
def project_trajectory_to_image(nusc, trajectory_world, camera_token):
    # ... 此函数代码与上一个回答中完全相同 ...
    cam_data = nusc.get('sample_data', camera_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
    points_ego = trajectory_world - np.array(pose_record['translation'])
    points_ego = np.dot(Quaternion(pose_record['rotation']).inverse.rotation_matrix, points_ego.T).T
    points_sensor = points_ego - np.array(cs_record['translation'])
    points_sensor = np.dot(Quaternion(cs_record['rotation']).inverse.rotation_matrix, points_sensor.T).T
    points_img = view_points(points_sensor.T, np.array(cs_record['camera_intrinsic']), normalize=True)
    projected_points = []
    for i in range(points_sensor.shape[0]):
        if points_sensor[i, 2] > 0.1:
            if 0 <= points_img[0, i] < cam_data['width'] and 0 <= points_img[1, i] < cam_data['height']:
                projected_points.append(points_img[:2, i].astype(int))
    return projected_points


def main():
    print("Initializing NuScenes...")
    nusc = NuScenes(version=CONFIG['version'], dataroot=CONFIG['dataroot'], verbose=False)

    print(f"Loading critical events from {CONFIG['critical_event_index_file']}...")
    with open(CONFIG['critical_event_index_file'], 'r') as f:
        critical_events = json.load(f)

    os.makedirs(CONFIG['visualization_output_dir'], exist_ok=True)
    total_events = sum(len(events) for events in critical_events.values())
    print(f"Found {len(critical_events)} scenes with a total of {total_events} events. Starting grid visualization...")
    event_progress = tqdm(total=total_events, desc="Processing Events")

    # 获取单个摄像头的尺寸用于创建网格
    any_scene = nusc.scene[0]
    any_sample = nusc.get('sample', any_scene['first_sample_token'])
    any_cam_token = any_sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', any_cam_token)
    cam_h, cam_w = cam_data['height'], cam_data['width']

    scaled_h, scaled_w = int(cam_h * CONFIG['grid_scale']), int(cam_w * CONFIG['grid_scale'])
    grid_h, grid_w = scaled_h * 2, scaled_w * 3  # 2行3列

    for scene_token, events in critical_events.items():
        scene = nusc.get('scene', scene_token)
        frame_idx_to_sample_token = {}
        current_sample_token = scene['first_sample_token']
        for i in range(scene['nbr_samples']):
            frame_idx_to_sample_token[i] = current_sample_token
            sample = nusc.get('sample', current_sample_token)
            current_sample_token = sample['next']
            if not current_sample_token: break

        for event_idx, event in enumerate(events):
            key_agent_token = event['instance_token']
            start_frame, end_frame = event['start_frame'], event['end_frame']

            event_folder_name = f"scene_{scene['name']}_event_{event_idx}_{event['reason']}"
            event_output_path = os.path.join(CONFIG['visualization_output_dir'], event_folder_name)
            os.makedirs(event_output_path, exist_ok=True)

            # --- 准备轨迹数据 (与之前相同) ---
            pred_future_world = np.array(event['predicted_trajectory'])
            pred_anchor_frame = event['peak_fde_frame_in_traj'] - 1
            gt_future_world = []
            instance_record = nusc.get('instance', key_agent_token)
            ann_token = instance_record['first_annotation_token']
            current_frame_idx = 0
            while ann_token:
                ann = nusc.get('sample_annotation', ann_token)
                if current_frame_idx >= pred_anchor_frame and len(gt_future_world) < (CONFIG['future_len'] + 1):
                    gt_future_world.append(ann['translation'][:2])
                ann_token = ann['next']
                current_frame_idx += 1
                if not ann_token: break
            gt_future_world = np.array(gt_future_world)

            # 补全Z轴坐标
            z_coord = nusc.get('sample_annotation', instance_record['first_annotation_token'])['translation'][2]

            # 对齐轨迹
            if len(gt_future_world) > 0:
                history_end_pos_gt = gt_future_world[0:1]
                pred_traj_to_draw = pred_future_world + (history_end_pos_gt - pred_future_world[0:1])
            else:  # 如果找不到GT轨迹，就使用预测轨迹的原点
                pred_traj_to_draw = pred_future_world

            gt_traj_3d = np.hstack([gt_future_world, np.full((gt_future_world.shape[0], 1), z_coord)])
            pred_traj_3d = np.hstack([pred_traj_to_draw, np.full((pred_traj_to_draw.shape[0], 1), z_coord)])

            for frame_idx in range(start_frame, end_frame):
                sample_token = frame_idx_to_sample_token.get(frame_idx)
                if not sample_token: continue

                # 创建一个黑色的大画布来拼接所有摄像头图像
                grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

                interacting_tokens = event.get('interactions', {}).get(str(frame_idx), [])
                sample_anns = nusc.get('sample', sample_token)['anns']

                for row_idx, row in enumerate(CONFIG['grid_layout']):
                    for col_idx, cam_name in enumerate(row):
                        camera_token = nusc.get('sample', sample_token)['data'][cam_name]

                        # --- 1. 获取原始图像 ---
                        img_path, _, cam_intrinsic = nusc.get_sample_data(camera_token,
                                                                          box_vis_level=BoxVisibility.NONE)
                        image = cv2.imread(img_path)
                        if image is None: continue

                        # --- 2. 在单个图像上渲染包围盒 ---
                        for ann_token in sample_anns:
                            ann = nusc.get('sample_annotation', ann_token)
                            if ann['instance_token'] != key_agent_token and ann[
                                'instance_token'] not in interacting_tokens:
                                continue  # 只渲染关键和交互agent

                            box = nusc.get_box(ann_token)
                            color = None
                            if ann['instance_token'] == key_agent_token:
                                color = CONFIG['color_key_agent_box']
                            elif ann['instance_token'] in interacting_tokens:
                                color = CONFIG['color_interaction_agent_box']

                            if color:
                                box.render_cv2(image, view=cam_intrinsic, normalize=True, colors=(color, color, color))

                        # --- 3. 在单个图像上渲染轨迹 ---
                        gt_points_2d = project_trajectory_to_image(nusc, gt_traj_3d, camera_token)
                        pred_points_2d = project_trajectory_to_image(nusc, pred_traj_3d, camera_token)
                        for i in range(len(gt_points_2d) - 1):
                            cv2.line(image, tuple(gt_points_2d[i]), tuple(gt_points_2d[i + 1]),
                                     CONFIG['color_gt_trajectory'], 2)
                        for i in range(len(pred_points_2d) - 1):
                            cv2.line(image, tuple(pred_points_2d[i]), tuple(pred_points_2d[i + 1]),
                                     CONFIG['color_pred_trajectory'], 2)

                        # --- 4. 缩放并粘贴到大画布上 ---
                        resized_image = cv2.resize(image, (scaled_w, scaled_h))

                        # 添加摄像头名称标签
                        cv2.putText(resized_image, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)

                        y_offset = row_idx * scaled_h
                        x_offset = col_idx * scaled_w
                        grid_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_image

                # 保存最终的网格图像
                output_filename = os.path.join(event_output_path, f"grid_frame_{frame_idx:03d}.jpg")
                cv2.imwrite(output_filename, grid_image)

            event_progress.update(1)

    event_progress.close()
    print("\nGrid visualization finished!")
    print(f"Results saved to: {CONFIG['visualization_output_dir']}")


if __name__ == '__main__':
    main()

