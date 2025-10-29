import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from rules_infer.tools.motion_lstm import *
import math
import json


def calculate_kinematics(positions, timestamps):
    """从位置和时间戳计算速度和加速度"""
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)

    dt = np.diff(timestamps)
    # 防止除以零
    dt[dt < 1e-6] = 1e-6

    # 计算速度
    vel = np.diff(positions, axis=0) / dt[:, np.newaxis]
    velocities[1:] = vel

    # 计算加速度
    accel = np.diff(vel, axis=0) / dt[1:, np.newaxis]
    accelerations[2:] = accel

    return velocities, accelerations


def check_line_segment_intersection(p1, p2, p3, p4):
    """检查线段(p1, p2)和(p3, p4)是否相交"""

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    return False


# ----------------- 主检测器类 -----------------

class SocialInteractionDetector:
    def __init__(self, model, nusc, config):
        self.model = model
        self.nusc = nusc
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def _get_agent_trajectories_in_scene(self, scene_token):
        """从nuScenes SDK中提取并整理场景中所有agent的轨迹数据"""
        scene = self.nusc.get('scene', scene_token)
        first_sample_token = scene['first_sample_token']

        agent_data = defaultdict(lambda: defaultdict(list))

        current_sample_token = first_sample_token
        while current_sample_token:
            sample = self.nusc.get('sample', current_sample_token)
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                instance_token = ann['instance_token']

                agent_data[instance_token]['timestamps'].append(sample['timestamp'])
                agent_data[instance_token]['positions'].append(ann['translation'][:2])  # 只取XY
                agent_data[instance_token]['rotations'].append(ann['rotation'])
                agent_data[instance_token]['category_name'].append(ann['category_name'])

            current_sample_token = sample['next']

        # 转换为numpy数组并计算运动学信息
        processed_agent_data = {}
        for instance_token, data in agent_data.items():
            if len(data['timestamps']) < self.config['HIST_LEN'] + self.config['PRED_LEN']:
                continue

            data_np = {
                'timestamps': np.array(data['timestamps']) / 1e6,  # 微秒转秒
                'positions': np.array(data['positions']),
                'rotations': data['rotations'],
                'category_name': data['category_name'][0]  # 假设类别不变
            }
            velocities, accelerations = calculate_kinematics(data_np['positions'], data_np['timestamps'])
            data_np['velocities'] = velocities
            data_np['accelerations'] = accelerations

            processed_agent_data[instance_token] = data_np

        return processed_agent_data

    def _calculate_trigger_metrics(self, pred_traj, gt_data):
        """计算事件触发指标"""
        gt_traj = gt_data['positions']

        # 1. 位移误差
        displacement_errors = np.linalg.norm(pred_traj - gt_traj, axis=1)
        ade = np.mean(displacement_errors)
        fde = displacement_errors[-1]

        # 2. 运动学误差
        # 为预测轨迹计算运动学
        pred_vel, pred_accel = calculate_kinematics(pred_traj, gt_data['timestamps'])

        # 提取真实轨迹的运动学
        gt_vel = gt_data['velocities']
        gt_accel = gt_data['accelerations']

        vel_error = np.linalg.norm(pred_vel - gt_vel, axis=1)
        accel_error = np.linalg.norm(pred_accel - gt_accel, axis=1)

        # 计算纵向和横向速度误差
        # 使用真实轨迹的朝向作为参考系
        long_vel_errors, lat_vel_errors = [], []
        for i in range(len(gt_data['rotations'])):
            q = Quaternion(gt_data['rotations'][i])
            heading_vec = q.rotation_matrix[:2, 0]  # X轴方向

            v_error_vec = gt_vel[i] - pred_vel[i]
            long_error = np.dot(v_error_vec, heading_vec)
            lat_error = np.linalg.norm(v_error_vec - long_error * heading_vec)

            long_vel_errors.append(long_error)
            lat_vel_errors.append(lat_error)

        return {
            "ADE": ade,
            "FDE": fde,
            "max_velocity_error": np.max(vel_error) if vel_error.size > 0 else 0,
            "max_acceleration_error": np.max(accel_error) if accel_error.size > 0 else 0,
            "max_longitudinal_velocity_error": np.max(np.abs(long_vel_errors)) if long_vel_errors else 0
        }

    def _calculate_attribution_metrics(self, primary_data, alter_data, event_start_idx, pred_len, primary_pred_traj):
        """计算交互归因指标"""
        H = self.config['HIST_LEN']
        event_end_idx = event_start_idx + pred_len

        # 找到两个agent在事件时间窗口内的重叠部分
        ts_primary_start = primary_data['timestamps'][event_start_idx]
        ts_primary_end = primary_data['timestamps'][event_end_idx - 1]

        alter_indices = np.where(
            (alter_data['timestamps'] >= ts_primary_start) &
            (alter_data['timestamps'] <= ts_primary_end)
        )[0]

        if len(alter_indices) == 0:
            return None

        # 1. 最小距离
        primary_pos_event = primary_data['positions'][event_start_idx:event_end_idx]
        alter_pos_event = alter_data['positions'][alter_indices]
        # 为了计算距离，我们需要对齐时间戳，这里做个简化，直接用所有组合的最小距离
        min_dist = np.min(
            np.linalg.norm(primary_pos_event[:, np.newaxis, :] - alter_pos_event[np.newaxis, :, :], axis=2))

        # 2. 最小TTC (在事件开始前计算)
        pre_event_idx_p = event_start_idx - 1
        pre_event_ts = primary_data['timestamps'][pre_event_idx_p]
        pre_event_idx_a = np.argmin(np.abs(alter_data['timestamps'] - pre_event_ts))

        pos_p = primary_data['positions'][pre_event_idx_p]
        vel_p = primary_data['velocities'][pre_event_idx_p]
        pos_a = alter_data['positions'][pre_event_idx_a]
        vel_a = alter_data['velocities'][pre_event_idx_a]

        rel_pos = pos_a - pos_p
        rel_vel = vel_a - vel_p

        dist_sq = np.dot(rel_pos, rel_pos)
        speed_sq = np.dot(rel_vel, rel_vel)

        min_ttc = float('inf')
        # 只有在相互靠近时才计算TTC
        if np.dot(rel_pos, rel_vel) < 0 and speed_sq > 1e-6:
            ttc = -np.dot(rel_pos, rel_vel) / speed_sq
            if ttc > 0:  # 只关心未来的碰撞
                # 简单TTC计算
                rel_speed_val = np.linalg.norm(vel_p - vel_a)
                if rel_speed_val > 0.1:  # 避免除以零
                    simple_ttc = np.linalg.norm(rel_pos) / rel_speed_val
                    min_ttc = simple_ttc

        # 3. 路径冲突分析
        path_conflict = False
        alter_path_segments = alter_data['positions'][alter_indices]
        for i in range(len(primary_pred_traj) - 1):
            p1, p2 = primary_pred_traj[i], primary_pred_traj[i + 1]
            for j in range(len(alter_path_segments) - 1):
                p3, p4 = alter_path_segments[j], alter_path_segments[j + 1]
                if check_line_segment_intersection(p1, p2, p3, p4):
                    path_conflict = True
                    break
            if path_conflict:
                break

        # 4. 相对位置
        # 使用Frenet坐标系太复杂，这里简化为"前方/后方/左侧/右侧"
        heading_q = Quaternion(primary_data['rotations'][pre_event_idx_p])
        heading_vec = heading_q.rotation_matrix[:2, 0]  # X-axis
        side_vec = heading_q.rotation_matrix[:2, 1]  # Y-axis

        long_dist = np.dot(rel_pos, heading_vec)
        lat_dist = np.dot(rel_pos, side_vec)

        if long_dist > 0:
            pos_desc = "front"
        else:
            pos_desc = "rear"

        if abs(lat_dist) > 2:  # 假设车道宽度约4米
            if lat_dist > 0:
                pos_desc += "-left"
            else:
                pos_desc += "-right"

        return {
            "min_distance": min_dist,
            "min_TTC_pre_event": min_ttc if min_ttc != float('inf') else None,
            "predicted_path_conflict": path_conflict,
            "relative_position": pos_desc
        }

    def analyze_scene(self, scene_token):
        """分析整个场景，检测并归因社会性交互事件"""
        all_agent_data = self._get_agent_trajectories_in_scene(scene_token)
        detected_events = []

        H = self.config['HIST_LEN']
        P = self.config['PRED_LEN']

        for primary_id, primary_data in all_agent_data.items():
            num_timesteps = len(primary_data['timestamps'])

            # 滑动窗口遍历轨迹
            for i in range(num_timesteps - H - P + 1):
                hist_end_idx = i + H

                # 1. 准备模型输入
                hist_positions = primary_data['positions'][i:hist_end_idx]
                # 归一化：减去最后一个历史点的位置
                last_hist_pos = hist_positions[-1]
                normalized_hist = hist_positions - last_hist_pos

                input_tensor = torch.from_numpy(normalized_hist).float().unsqueeze(0).to(self.device)

                # 2. 模型预测
                with torch.no_grad():
                    pred_relative_traj = self.model(input_tensor).squeeze(0).cpu().numpy()

                # 反归一化
                pred_abs_traj = pred_relative_traj + last_hist_pos

                # 3. 准备真值数据
                gt_start_idx = hist_end_idx
                gt_end_idx = gt_start_idx + P
                gt_data_slice = {key: val[gt_start_idx:gt_end_idx] for key, val in primary_data.items()}

                # 4. 计算触发指标
                trigger_metrics = self._calculate_trigger_metrics(pred_abs_traj, gt_data_slice)

                # 5. 判断是否触发事件
                is_event = (trigger_metrics['FDE'] > self.config['THRESHOLDS']['FDE'] or
                            trigger_metrics['max_longitudinal_velocity_error'] > self.config['THRESHOLDS'][
                                'LONG_VEL_ERROR'])

                if is_event:
                    # 6. 如果触发，开始归因分析
                    candidate_agents = []
                    for alter_id, alter_data in all_agent_data.items():
                        if alter_id == primary_id:
                            continue

                        attribution_metrics = self._calculate_attribution_metrics(primary_data, alter_data,
                                                                                  gt_start_idx, P, pred_abs_traj)
                        if attribution_metrics:
                            # 简单交互评分：距离越近、TTC越小，分数越高
                            score = 0
                            if attribution_metrics['min_distance'] < 10:  # 只考虑10米内的agent
                                score += 1.0 / (attribution_metrics['min_distance'] + 1e-6)
                                if attribution_metrics['min_TTC_pre_event'] and attribution_metrics[
                                    'min_TTC_pre_event'] < 5:
                                    score += 2.0 / (attribution_metrics['min_TTC_pre_event'] + 1e-6)
                                if attribution_metrics['predicted_path_conflict']:
                                    score += 5.0  # 路径冲突是强信号

                            if score > 0.1:  # 过滤掉不相关的agent
                                candidate_agents.append({
                                    "agent_id": alter_id,
                                    "agent_type": alter_data['category_name'],
                                    "interaction_score": score,
                                    "attribution_metrics": attribution_metrics
                                })

                    if not candidate_agents:  # 如果没有找到交互对象，可能不是交互事件
                        continue

                    candidate_agents.sort(key=lambda x: x['interaction_score'], reverse=True)

                    event_report = {
                        "event_id": f"{scene_token}_{primary_id[:8]}_{gt_start_idx}",
                        "timestamp_start": primary_data['timestamps'][gt_start_idx],
                        "timestamp_end": primary_data['timestamps'][gt_end_idx - 1],
                        "scene_context": {
                            "location": self.nusc.get('log', self.nusc.get('scene', scene_token)['log_token'])[
                                'location'],
                            # HD Map信息需要更复杂的API调用，这里暂时留空
                            "map_elements": []
                        },
                        "primary_agent": {
                            "agent_id": primary_id,
                            "agent_type": primary_data['category_name'],
                            "event_trigger_metrics": trigger_metrics,
                            "behavior_summary": "Deviation from kinematic prediction detected."
                        },
                        "candidate_interacting_agents": candidate_agents
                    }
                    detected_events.append(event_report)

        return detected_events


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. 配置参数
    CONFIG = {
        'HIST_LEN': 10,  # 历史轨迹长度 (2秒 @ 5Hz)
        'PRED_LEN': 12,  # 预测轨迹长度 (2.4秒 @ 5Hz)
        'INPUT_DIM': 2,  # 输入维度 (x, y)
        'OUTPUT_DIM': 2,  # 输出维度 (x, y)
        'THRESHOLDS': {
            'FDE': 3.0,  # Final Displacement Error 阈值 (米)
            'LONG_VEL_ERROR': 2.0  # 纵向速度误差阈值 (米/秒), 表示明显的加减速
        }
    }

    OUTPUT_JSON_PATH = "result.json"
    # 2. 初始化nuScenes SDK
    # 请修改为你自己的路径
    nusc = NuScenes(version='v1.0-trainval', dataroot='/data0/senzeyu2/dataset/nuscenes', verbose=True)

    # 3. 加载你预训练的LSTM模型
    # 这里我们创建一个假的模型并加载一个虚拟的权重，你需要替换成你自己的
    model = TrajectoryLSTM(CONFIG)
    try:
        # 替换 'trajectory_lstm.pth' 为你的模型文件
        model.load_state_dict(torch.load('trajectory_lstm.pth', map_location=device))
        print("Successfully loaded pre-trained model.")
    except FileNotFoundError:
        print("Warning: Model file 'trajectory_lstm.pth' not found. Using a randomly initialized model.")
        # 如果没有模型，代码也能运行，但预测会是随机的，可能检测出很多“假”事件

    # 4. 创建并运行检测器
    detector = SocialInteractionDetector(model, nusc, CONFIG)

    # 以nuScenes-mini中的一个经典交互场景为例：scene-0103
    # 这个场景中，ego-vehicle在一个T字路口礼让行人
    test_scene_token = nusc.scene[10]['token']  # scene-0103
    print(f"\nAnalyzing scene: {nusc.get('scene', test_scene_token)['name']} ({test_scene_token})")

    social_events = detector.analyze_scene(test_scene_token)

    # 5. 打印结果
    if social_events:
        print(f"\nDetected {len(social_events)} potential social interaction events.")
        # 只打印第一个事件的详细信息作为示例
        first_event = social_events[0]
        print("\n--- Example Event Report ---")
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(first_event, f, indent=2,
                         default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else o)
    else:
        print("\nNo significant social interaction events detected in this scene.")
