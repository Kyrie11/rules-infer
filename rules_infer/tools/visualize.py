import json
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# ------------------- CONFIGURATION -------------------
NUSCENES_DATAROOT = '/data0/senzeyu2/dataset/nuscenes/'  # <-- 修改为你的 NuScenes 路径
NUSCENES_VERSION = 'v1.0-mini'  # <-- 建议先用 mini 测试
EVENTS_FILE = 'critical_events.json'  # <-- 你的事件文件
OUTPUT_DIR = 'visualization_output'  # <-- 保存可视化结果的文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 可视化颜色 (BGR format for OpenCV) ---
COLOR_KEY_AGENT = (0, 0, 255)  # Red
COLOR_INTERACTING = (255, 0, 0)  # Blue
COLOR_GT_TRAJ = (0, 255, 0)  # Green
COLOR_PRED_TRAJ = (0, 0, 255)  # Red


# ------------------- HELPER FUNCTIONS -------------------

def get_full_trajectories_for_scene(nusc, scene_token):
    """
    为了效率，一次性加载一个场景中所有agent的完整轨迹。
    返回: {instance_token: np.array([x, y, z]), ...}
    """
    scene_rec = nusc.get('scene', scene_token)

    # 获取场景中所有 annotation tokens
    ann_tokens = nusc.field2token('sample_annotation', 'scene_token', scene_token)

    # 按 instance_token 分组
    instance_to_anns = defaultdict(list)
    for ann_token in ann_tokens:
        ann = nusc.get('sample_annotation', ann_token)
        instance_to_anns[ann['instance_token']].append(ann)

    # 排序并提取轨迹
    full_trajectories = {}
    for instance_token, anns in instance_to_anns.items():
        # 按时间戳排序
        anns.sort(key=lambda x: nusc.get('sample', x['sample_token'])['timestamp'])

        trajectory = np.array([ann['translation'] for ann in anns])
        full_trajectories[instance_token] = trajectory

    return full_trajectories


def map_pointcloud_to_image_custom(points, camera_intrinsic, camera_extrinsic):
    """
    一个辅助函数，用于将3D点云投影到2D图像平面。
    points: 3xN 的点云矩阵
    camera_intrinsic: 3x3 的相机内参矩阵
    camera_extrinsic: 包含 'translation' 和 'rotation' 的记录
    """
    # 转换到 ego vehicle frame
    points = points - np.array(camera_extrinsic['translation']).reshape((-1, 1))
    points = np.dot(Quaternion(camera_extrinsic['rotation']).inverse.rotation_matrix, points)

    # 转换到 camera frame
    # (NuScenes 相机坐标系和车辆坐标系可能不同，但这里简化处理)

    # 投影到图像平面
    points_img = np.dot(camera_intrinsic, points)

    # 深度归一化
    points_img[:2, :] /= points_img[2, :]

    return points_img[:2, :]


# ------------------- MAIN SCRIPT -------------------

if __name__ == "__main__":
    # 1. 初始化 NuScenes
    print("Initializing NuScenes...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)

    # 2. 加载事件文件
    print(f"Loading events from {EVENTS_FILE}...")
    with open(EVENTS_FILE, 'r') as f:
        critical_events = json.load(f)

    # 3. 遍历和处理每个事件
    for scene_token, events_in_scene in critical_events.items():
        print(f"\nProcessing Scene: {scene_token}")

        # 预加载该场景的所有轨迹数据和 sample tokens
        scene_trajectories = get_full_trajectories_for_scene(nusc, scene_token)
        scene_sample_tokens = nusc.get_sample_tokens_in_scene(scene_token)

        for i, event in enumerate(events_in_scene):
            key_agent_token = event['instance_token']
            case_id = f"{scene_token}_{key_agent_token}_{i}"
            event_output_dir = os.path.join(OUTPUT_DIR, case_id)
            os.makedirs(event_output_dir, exist_ok=True)

            print(f"  - Visualizing event {case_id} from frame {event['start_frame']} to {event['end_frame']}")

            # 遍历事件窗口中的每一帧
            for frame_idx in tqdm(range(event['start_frame'], event['end_frame']), desc=f"    Frames for event {i}"):
                if frame_idx >= len(scene_sample_tokens):
                    continue

                sample_token = scene_sample_tokens[frame_idx]
                sample_rec = nusc.get('sample', sample_token)

                # 我们只选择前置摄像头进行可视化，以简化问题
                cam_token = sample_rec['data']['CAM_FRONT']
                sd_rec = nusc.get('sample_data', cam_token)
                cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

                image_path = os.path.join(nusc.dataroot, sd_rec['filename'])
                image = cv2.imread(image_path)

                # --- 绘制边界框 ---
                interacting_tokens = event.get('interactions', {}).get(str(frame_idx), [])

                for agent_token in [key_agent_token] + interacting_tokens:
                    color = COLOR_KEY_AGENT if agent_token == key_agent_token else COLOR_INTERACTING
                    try:
                        # 找到该agent在当前sample中的annotation
                        ann_token = nusc.get_sample_annotation_token(sample_token, agent_token)
                        ann_rec = nusc.get('sample_annotation', ann_token)

                        box = Box(ann_rec['translation'], ann_rec['size'], Quaternion(ann_rec['rotation']))

                        # 使用NuScenes的工具函数渲染边界框
                        box.render_cv2(image, view=np.array(cs_rec['camera_intrinsic']), normalize=True,
                                       cs_record=cs_rec, sd_record=sd_rec, color=color, thickness=2)
                    except KeyError:
                        # Agent可能在这一帧暂时消失或不存在
                        continue

                # --- 绘制轨迹 (只在事件中心帧绘制) ---
                if frame_idx == event['peak_fde_frame_in_traj']:
                    # 1. 真实轨迹 (从完整的轨迹数据中获取)
                    # GT 轨迹是从 t0+1 到 t0+future_len
                    gt_start_idx = frame_idx + 1
                    gt_end_idx = frame_idx + 1 + 12  # 假设 future_len = 12

                    if key_agent_token in scene_trajectories:
                        gt_traj_world = scene_trajectories[key_agent_token][gt_start_idx:gt_end_idx]

                        if gt_traj_world.shape[0] > 1:
                            # 投影到图像
                            points = gt_traj_world.T  # shape (3, N)
                            points_cam = map_pointcloud_to_image_custom(points, np.array(cs_rec['camera_intrinsic']),
                                                                        cs_rec)
                            points_2d = points_cam.T.astype(np.int32)
                            cv2.polylines(image, [points_2d], isClosed=False, color=COLOR_GT_TRAJ, thickness=3)

                    # 2. 预测轨迹 (从 event 文件中获取)
                    pred_traj_world = np.array(event['predicted_trajectory'])
                    if pred_traj_world.shape[0] > 1:
                        # 投影到图像
                        points_3d = np.hstack(
                            [pred_traj_world, np.ones((pred_traj_world.shape[0], 1)) * 1.5]).T  # 假设高度为1.5m
                        points_cam = map_pointcloud_to_image_custom(points_3d, np.array(cs_rec['camera_intrinsic']),
                                                                    cs_rec)
                        points_2d = points_cam.T.astype(np.int32)
                        cv2.polylines(image, [points_2d], isClosed=False, color=COLOR_PRED_TRAJ, thickness=3,
                                      lineType=cv2.LINE_AA)

                # --- 保存图像 ---
                output_image_path = os.path.join(event_output_dir, f"
