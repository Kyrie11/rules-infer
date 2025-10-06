import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from itertools import combinations


def calculate_lstm_surprise(model, instance_history, history_len=10, future_len=15):
    """使用训练好的LSTM模型计算惊讶度"""
    model.eval()
    device = next(model.parameters()).device
    surprise_scores = {}
    total_len = history_len + future_len

    if len(instance_history) < total_len:
        return {}

    with torch.no_grad():
        # 滑动窗口
        for i in range(len(instance_history) - total_len + 1):
            seq_data = instance_history[i: i + total_len]

            # 与数据准备时完全相同的特征提取过程
            anchor_ann = seq_data[history_len - 1]['ann']
            anchor_pos = np.array(anchor_ann['translation'])[:2]
            anchor_quat = Quaternion(anchor_ann['rotation'])

            hist_features_list = []
            for j in range(history_len):
                ann = seq_data[j]['ann']
                pos = np.array(ann['translation'])[:2];
                quat = Quaternion(ann['rotation'])
                relative_pos = pos - anchor_pos
                rotated_pos = anchor_quat.inverse.rotate(np.array([*relative_pos, 0.0]))
                relative_yaw = (quat * anchor_quat.inverse).yaw_pitch_roll[0]
                hist_features_list.append([rotated_pos[0], rotated_pos[1], np.cos(relative_yaw), np.sin(relative_yaw)])

            history_tensor = torch.FloatTensor([hist_features_list]).to(device)

            # 模型预测 (局部坐标系)
            pred_future_local = model(history_tensor).squeeze(0).cpu().numpy()

            # 获取真实未来轨迹 (局部坐标系)
            gt_future_local = []
            for j in range(history_len, total_len):
                ann = seq_data[j]['ann']
                pos = np.array(ann['translation'])[:2]
                relative_pos = pos - anchor_pos
                rotated_pos = anchor_quat.inverse.rotate(np.array([*relative_pos, 0.0]))
                gt_future_local.append(rotated_pos[:2])
            gt_future_local = np.array(gt_future_local)

            # 计算FDE (Final Displacement Error) 作为惊讶度分数
            fde = np.linalg.norm(pred_future_local[-1] - gt_future_local[-1])

            # 记录在当前帧 (即历史的最后一帧)
            sample_token = anchor_ann['sample_token']
            surprise_scores[sample_token] = float(fde)

    return surprise_scores

# --- 动态计算函数 (无变化) ---
def get_instance_record(nusc: NuScenes, instance_token: str, sample_token: str):
    for ann_token in nusc.get('sample', sample_token)['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token:
            return ann
    return None


def calculate_agent_dynamics(nusc: NuScenes, sample_ann: dict):
    try:
        velocity = nusc.box_velocity(sample_ann['token'])[:2]
        if np.isnan(velocity).any(): velocity = np.array([0.0, 0.0])
    except ValueError:
        velocity = np.array([0.0, 0.0])

    acceleration = np.array([0.0, 0.0])
    yaw_rate = 0.0

    if sample_ann['prev']:
        prev_ann = nusc.get('sample_annotation', sample_ann['prev'])
        curr_sample = nusc.get('sample', sample_ann['sample_token'])
        prev_sample = nusc.get('sample', prev_ann['sample_token'])
        time_diff = (curr_sample['timestamp'] - prev_sample['timestamp']) * 1e-6

        if time_diff > 0.01:
            try:
                prev_velocity = nusc.box_velocity(prev_ann['token'])[:2]
                if np.isnan(prev_velocity).any(): prev_velocity = np.array([0.0, 0.0])
            except ValueError:
                prev_velocity = np.array([0.0, 0.0])
            acceleration = (velocity - prev_velocity) / time_diff
            curr_quat = Quaternion(sample_ann['rotation'])
            prev_quat = Quaternion(prev_ann['rotation'])
            delta_quat = curr_quat * prev_quat.inverse
            yaw_rate = delta_quat.angle / time_diff if delta_quat.angle is not None else 0.0
            if delta_quat.axis[2] < 0:
                yaw_rate = -yaw_rate

    return velocity, acceleration, yaw_rate

def find_key_event_in_scene(nusc: NuScenes, scene_token: str, weights: dict = None):
    if weights is None:
        weights = {'accel_lon': 1.0, 'accel_lat': 1.5, 'yaw_rate': 2.0}
    scene = nusc.get('scene', scene_token)
    max_score = -1
    key_sample_token = None
    protagonist_instance_token = None
    current_sample_token = scene['first_sample_token']
    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            if 'vehicle' not in ann['category_name'] and 'human' not in ann['category_name']:
                continue
            velocity, acceleration, yaw_rate = calculate_agent_dynamics(nusc, ann)
            q = Quaternion(ann['rotation'])
            heading_vector = q.rotate(np.array([1.0, 0, 0]))[:2]
            if np.linalg.norm(heading_vector) > 0:
                accel_lon = np.dot(acceleration, heading_vector) / np.linalg.norm(heading_vector)
                accel_lat = np.cross(np.array([*heading_vector, 0]), np.array([*acceleration, 0]))[2] / np.linalg.norm(heading_vector)
            else:
                accel_lon, accel_lat = 0.0, 0.0
            instability_score = (abs(accel_lon) * weights['accel_lon'] +
                                 abs(accel_lat) * weights['accel_lat'] +
                                 abs(yaw_rate) * weights['yaw_rate'])
            if instability_score > max_score:
                max_score = instability_score
                key_sample_token = sample['token']
                protagonist_instance_token = ann['instance_token']
        current_sample_token = sample['next']
    return key_sample_token, protagonist_instance_token, max_score

def get_clip_tokens(nusc: NuScenes, scene_token: str, center_sample_token: str, clip_duration_s: float = 4.0):
    scene = nusc.get('scene', scene_token)
    samples = []
    curr_token = scene['first_sample_token']
    while curr_token:
        samples.append(nusc.get('sample', curr_token))
        curr_token = samples[-1]['next']
    center_idx = [i for i, s in enumerate(samples) if s['token'] == center_sample_token][0]
    num_frames_half = int(clip_duration_s / 2 * 2)
    start_idx = max(0, center_idx - num_frames_half)
    end_idx = min(len(samples), center_idx + num_frames_half + 1)
    clip_samples = samples[start_idx:end_idx]
    return [s['token'] for s in clip_samples]


# --- 新增: 关系计算辅助函数 ---
def calculate_relative_kinematics(from_ann: dict, to_ann: dict, from_vel: np.ndarray, to_vel: np.ndarray):
    """
    计算从 'from_ann' 视角看 'to_ann' 的相对运动学信息。
    """
    from_pos = np.array(from_ann['translation'])
    to_pos = np.array(to_ann['translation'])
    from_quat = Quaternion(from_ann['rotation'])

    # 1. 距离
    distance = np.linalg.norm(to_pos - from_pos)

    # 2. 相对速度 (全局坐标系)
    relative_velocity_global = to_vel - from_vel

    # 3. 相对位置 (在 'from' 的局部坐标系中)
    global_diff_vec = to_pos - from_pos
    relative_position_local = from_quat.inverse.rotate(global_diff_vec)

    return {
        "distance": float(distance),
        "relative_position": list(relative_position_local),
        "relative_velocity_global": list(relative_velocity_global)
    }


# --- 核心修改: 拓扑图构建函数 ---
def build_event_topology_graph(nusc: NuScenes, sample_token: str, protagonist_token: str):
    """
    为单个sample构建包含丰富细节和二级交互的拓扑图。
    """
    sample = nusc.get('sample', sample_token)
    nodes, edges = [], []

    # 1. 缓存所有agent的annotation和动态信息
    agent_anns = {ann['instance_token']: ann for ann in [nusc.get('sample_annotation', t) for t in sample['anns']]}
    agent_dynamics = {token: calculate_agent_dynamics(nusc, ann) for token, ann in agent_anns.items()}

    # 2. 创建节点
    ego_pose_rec = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
    ego_ann_like = {"translation": ego_pose_rec['translation'], "rotation": ego_pose_rec['rotation']}
    nodes.append({
        "id": "ego", "category": "vehicle.car", "is_protagonist": False, "is_observer": True,
        "state": {"position": ego_ann_like['translation'], "rotation": ego_ann_like['rotation']}
    })

    for token, ann in agent_anns.items():
        vel, acc, yaw_r = agent_dynamics[token]
        nodes.append({
            "id": token, "category": ann['category_name'], "is_protagonist": token == protagonist_token,
            "is_observer": False,
            "state": {
                "position": ann['translation'], "rotation": ann['rotation'],
                "velocity": list(vel), "acceleration": list(acc), "yaw_rate": yaw_r
            }
        })

    # 3. 构建边
    protagonist_ann = agent_anns.get(protagonist_token)
    if not protagonist_ann: return {"nodes": nodes, "edges": []}

    protagonist_vel = agent_dynamics[protagonist_token][0]
    significant_actors = set()

    # --- 优先级1: 主角 -> 配角 ---
    for other_token, other_ann in agent_anns.items():
        if other_token == protagonist_token: continue

        other_vel = agent_dynamics[other_token][0]
        kinematics = calculate_relative_kinematics(protagonist_ann, other_ann, protagonist_vel, other_vel)

        is_significant = False
        if kinematics['distance'] < 25.0:  # 距离阈值
            if kinematics['relative_position'][0] > -2.0:  # 在主角前方或侧方，而非纯后方
                is_significant = True
            if np.linalg.norm(kinematics['relative_velocity_global']) > 5.0:
                is_significant = True

        if is_significant:
            significant_actors.add(other_token)
            edges.append({
                "from": protagonist_token, "to": other_token,
                "relation_type": "significant_interaction", "details": kinematics
            })

    # --- 优先级2: 配角 <-> 配角 ---
    if len(significant_actors) > 1:
        for token_a, token_b in combinations(significant_actors, 2):
            ann_a, ann_b = agent_anns[token_a], agent_anns[token_b]
            vel_a, vel_b = agent_dynamics[token_a][0], agent_dynamics[token_b][0]

            kinematics_ab = calculate_relative_kinematics(ann_a, ann_b, vel_a, vel_b)

            # 配角间的交互判断可以稍微宽松一些，只看距离
            if kinematics_ab['distance'] < 20.0:
                edges.append({
                    "from": token_a, "to": token_b,
                    "relation_type": "secondary_interaction", "details": kinematics_ab
                })

    # --- 优先级3: Ego -> 关键角色 ---
    ego_vel = np.array([0, 0])  # 假设Ego速度信息不易获取，或设为0
    all_key_actors = {protagonist_token}.union(significant_actors)
    for actor_token in all_key_actors:
        if actor_token in agent_anns:
            actor_ann = agent_anns[actor_token]
            actor_vel = agent_dynamics[actor_token][0]
            kinematics_ego = calculate_relative_kinematics(ego_ann_like, actor_ann, ego_vel, actor_vel)
            edges.append({
                "from": "ego", "to": actor_token,
                "relation_type": "observing", "details": kinematics_ego
            })

    return {"nodes": nodes, "edges": edges}


# --- 主执行流程 (无变化) ---
if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/senzeyu2/dataset/nuscenes', verbose=True)

    scene_token = nusc.scene[0]['token']
    print(f"Analyzing scene: {nusc.get('scene', scene_token)['name']}")

    key_sample_token, protagonist_token, max_score = find_key_event_in_scene(nusc, scene_token)

    if not key_sample_token:
        print("No significant event found in this scene.")
    else:
        print(f"\n--- Key Event Found ---")
        print(f"Time (Sample Token): {key_sample_token}")
        print(f"Protagonist (Instance Token): {protagonist_token}")
        print(f"Max Instability Score: {max_score:.2f}")

        clip_sample_tokens = get_clip_tokens(nusc, scene_token, key_sample_token, clip_duration_s=4.0)
        print(f"\nExtracted a clip of {len(clip_sample_tokens)} frames around the event.")

        full_clip_data_for_vlm = []
        for i, sample_token in enumerate(clip_sample_tokens):
            print(f"  Processing frame {i + 1}/{len(clip_sample_tokens)}...")
            topology_graph = build_event_topology_graph(nusc, sample_token, protagonist_token)
            full_clip_data_for_vlm.append({
                "frame_index": i, "sample_token": sample_token, "topology": topology_graph
            })

        print("\n--- Topology Graph for the event's peak frame ---")
        # 找到事件高潮帧在clip中的数据并打印
        peak_frame_data = next((frame for frame in full_clip_data_for_vlm if frame['sample_token'] == key_sample_token),
                               None)
        if peak_frame_data:
            print(json.dumps(peak_frame_data['topology'], indent=2))
