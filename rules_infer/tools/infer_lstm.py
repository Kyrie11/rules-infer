import torch
from torch.utils.data import DataLoader, random_split
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
# 确保 nuscenes-devkit 和其他库已安装
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from rules_infer.tools.motion_lstm import *
import math


class Config:
    # --- 数据集与路径 ---
    # !!! 修改为你的nuScenes数据集根目录 !!!
    NUSCENES_DATA_ROOT = '/data0/senzeyu2/dataset/nuscenes'
    NUSCENES_VERSION = 'v1.0-trainval'  # 使用mini数据集进行快速演示

    # --- 模型与训练参数 ---
    HIST_LEN = 8  # 历史轨迹长度 (N_in)
    PRED_LEN = 12  # 预测轨迹长度 (N_out)
    INPUT_DIM = 2  # 输入特征维度 (x, y)
    OUTPUT_DIM = 2  # 输出特征维度 (x, y)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20  # 演示目的，实际可增加
    MODEL_PATH = 'trajectory_lstm.pth'

    # --- 事件检测与分析参数 ---
    FDE_THRESHOLD_M = 2.0  # 最终位移误差的绝对阈值（米）
    FDE_VEL_MULTIPLIER = 1.5  # FDE的相对阈值，FDE > 速度 * 这个乘数
    TTC_THRESHOLD_S = 4.0  # 触发交互分析的碰撞时间阈值（秒）


class EventAnalyzer:
    """
    对单个潜在事件进行深入分析，包括运动学(L2)和关系(L3)指标。
    这个类完全基于预先加载的 scene_cache 工作，不直接调用 nuscenes API。
    """

    def __init__(self, config, scene_cache, primary_agent_token, event_timestamp, all_predictions):
        self.config = config
        self.scene_cache = scene_cache
        self.primary_agent_token = primary_agent_token
        self.event_timestamp = event_timestamp
        self.all_predictions = all_predictions

        self.primary_data = self.scene_cache['agents'].get(primary_agent_token, {})
        self.primary_track = self.primary_data.get('track', [])
        self.primary_state_at_event = self._get_state_from_track(self.primary_track, self.event_timestamp)

    def analyze(self):
        """执行完整的事件分析并返回结构化的事件数据，如果事件不显著则返回None。"""
        if not self.primary_state_at_event:
            return None

        # --- L2: 定性指标分析 (自身运动学) ---
        kinematics = self._calculate_kinematics()

        # --- L3: 关系指标分析 (与他车交互) ---
        interacting_agents, relational_metrics = self._analyze_relations()

        # --- 综合判断与打标签 ---
        event_label = self._determine_event_label(kinematics, relational_metrics)

        # 过滤掉不够显著的事件
        if event_label == "High_FDE_Only":
            return None

        # --- 整理并返回结果 ---
        event_data = {
            "scene_token": self.scene_cache['scene_token'],
            "start_timestamp": self.event_timestamp,
            "end_timestamp": self.event_timestamp + int(self.config.PRED_LEN * self.config.SAMPLE_INTERVAL_S * 1e6),
            "primary_agent_id": self.primary_agent_token,
            "primary_agent_category": self.primary_data.get('category', 'unknown'),
            "interacting_agent_ids": [agent['token'] for agent in interacting_agents],
            "event_type_label": event_label,
            "metrics_snapshot": {
                "fde": relational_metrics.get('fde'),
                "primary_agent_speed_kmh": self.primary_state_at_event['speed'] * 3.6,
                "max_abs_long_accel": kinematics.get('long_accel'),
                "max_abs_lat_accel": kinematics.get('lat_accel'),
                "max_abs_jerk": kinematics.get('jerk'),
                "max_abs_yaw_rate_dps": math.degrees(kinematics.get('yaw_rate', 0)),
                "min_ttc_s": relational_metrics.get('min_ttc'),
                "min_thw_s": relational_metrics.get('min_thw'),
            }
        }
        return event_data

    def _get_state_from_track(self, track, timestamp):
        """内部辅助函数：从轨迹列表中安全地查找特定时间戳的状态。"""
        return next((state for state in track if state['timestamp'] == timestamp), None)

    def _calculate_kinematics(self):
        """计算加速度、Jerk和偏航率。"""
        kinematics = {}

        try:
            event_idx = next(
                i for i, state in enumerate(self.primary_track) if state['timestamp'] == self.event_timestamp)
        except StopIteration:
            return kinematics

        win = self.config.KINEMATICS_WINDOW_SIZE
        if event_idx < win // 2 or event_idx >= len(self.primary_track) - (win // 2):
            return kinematics

        segment = self.primary_track[event_idx - win // 2: event_idx + win // 2 + 1]

        pos = np.array([s['pos'] for s in segment])
        ts = np.array([s['timestamp'] for s in segment]) / 1e6
        dt = np.diff(ts)

        # 确保 dt 不为零
        dt[dt < 1e-6] = 1e-6

        vel = np.diff(pos, axis=0) / dt[:, np.newaxis]
        accel = np.diff(vel, axis=0) / dt[1:, np.newaxis]
        jerk = np.diff(accel, axis=0) / dt[2:, np.newaxis] if len(dt) > 2 else np.array([])

        headings = np.unwrap([s['heading'] for s in segment])
        yaw_rate = np.diff(headings) / dt

        center_idx = win // 2
        heading_vec = np.array([math.cos(segment[center_idx]['heading']), math.sin(segment[center_idx]['heading'])])
        center_accel = accel[center_idx - 1]

        kinematics['long_accel'] = np.dot(center_accel, heading_vec)
        kinematics['lat_accel'] = np.cross(heading_vec, center_accel)
        kinematics['jerk'] = np.linalg.norm(jerk[center_idx - 2]) if jerk.size > 0 else 0
        kinematics['yaw_rate'] = yaw_rate[center_idx - 1]

        return kinematics

    def _analyze_relations(self):
        """分析与其他agent的关系：TTC, THW, Zone of Influence。"""
        interacting_agents = []
        relational_metrics = {}

        # 获取此事件的FDE
        agent_predictions = self.all_predictions.get(self.primary_agent_token, [])
        pred_info = next((p for p in agent_predictions if p['timestamp'] == self.event_timestamp), None)
        if pred_info:
            relational_metrics['fde'] = np.linalg.norm(pred_info['ground_truth'][-1] - pred_info['predicted'][-1])

        min_ttc, min_thw = float('inf'), float('inf')
        ttc_agent_info, thw_agent_info = None, None

        primary_pos = self.primary_state_at_event['pos']
        primary_vel = self.primary_state_at_event['vel']
        primary_heading = self.primary_state_at_event['heading']
        primary_speed = self.primary_state_at_event['speed']

        for other_token, other_data in self.scene_cache['agents'].items():
            if other_token == self.primary_agent_token:
                continue

            other_state = self._get_state_from_track(other_data.get('track', []), self.event_timestamp)
            if not other_state:
                continue

            rel_pos = other_state['pos'] - primary_pos

            q_inv = Quaternion(axis=[0, 0, 1], angle=-primary_heading)
            rel_pos_local = q_inv.rotate(np.array([rel_pos[0], rel_pos[1], 0]))[:2]

            is_in_zone = (-self.config.INTERACTION_ZONE_BACKWARD < rel_pos_local[
                0] < self.config.INTERACTION_ZONE_FORWARD and abs(
                rel_pos_local[1]) < self.config.INTERACTION_ZONE_LATERAL)
            if not is_in_zone: continue

            # 计算TTC
            rel_vel = primary_vel - other_state['vel']
            rel_speed = np.linalg.norm(rel_vel)
            if rel_speed > 1e-6 and np.dot(rel_pos, rel_vel) < 0:  # 正在接近
                ttc = np.linalg.norm(rel_pos) / rel_speed
                if ttc < min_ttc:
                    min_ttc = ttc
                    ttc_agent_info = {'token': other_token, 'state': other_state}

            # 计算THW
            if rel_pos_local[0] > 0 and primary_speed > 1.0:  # 对方在前方且自己在移动
                dist = np.linalg.norm(rel_pos) - 2.0  # 减去一个车长近似值
                thw = max(0, dist) / primary_speed
                if thw < min_thw:
                    min_thw = thw
                    thw_agent_info = {'token': other_token, 'state': other_state}

        if min_ttc < self.config.TTC_THRESHOLD:
            relational_metrics['min_ttc'] = min_ttc
            if ttc_agent_info and ttc_agent_info not in interacting_agents:
                interacting_agents.append(ttc_agent_info)

        if min_thw < self.config.THW_THRESHOLD_FOLLOW:
            relational_metrics['min_thw'] = min_thw
            if thw_agent_info and thw_agent_info not in interacting_agents:
                interacting_agents.append(thw_agent_info)

        return interacting_agents, relational_metrics

    def _determine_event_label(self, kinematics, relational_metrics):
        """基于所有指标，给事件打上一个描述性标签。"""
        jerk = kinematics.get('jerk', 0)
        long_accel = kinematics.get('long_accel', 0)
        yaw_rate = abs(kinematics.get('yaw_rate', 0))
        min_ttc = relational_metrics.get('min_ttc', float('inf'))

        is_hard_brake = long_accel < -3.0
        is_sudden_jerk = jerk > self.config.JERK_THRESHOLD
        is_sudden_swerve = yaw_rate > self.config.YAW_RATE_THRESHOLD
        is_ttc_critical = min_ttc < self.config.TTC_THRESHOLD

        if is_hard_brake or is_sudden_jerk:
            if is_ttc_critical:
                return "CRITICAL_BRAKE_FOR_LEAD_AGENT"
            else:
                return "SUDDEN_BRAKE_UNEXPLAINED"

        if is_sudden_swerve:
            if is_ttc_critical:
                return "EVASIVE_SWERVE_FOR_AGENT"
            else:
                return "ABRUPT_LANE_CHANGE_OR_SWERVE"

        if self.primary_state_at_event['speed'] < 1.0 and relational_metrics.get('fde', 0) > 1.5:
            if is_ttc_critical:
                return "YIELDING_AT_INTERSECTION_OR_MERGE"

        return "High_FDE_Only"


def build_scene_track_cache(nusc, scene):
    """
    为单个场景构建所有agent的轨迹缓存。
    这个函数是关键，它将一次性提取所有需要的信息。
    """
    cache = {'scene_token': scene['token'], 'agents': {}}
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    while sample:
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            # 筛选我们关心的agent类别
            category = ann['category_name']
            if not any(cat in category for cat in ['vehicle', 'human.pedestrian', 'bicycle']):
                continue

            inst_token = ann['instance_token']
            if inst_token not in cache['agents']:
                cache['agents'][inst_token] = {
                    'track': [],
                    'category': category  # 记录agent的类别
                }

            # --- 核心修正部分 ---
            # 直接使用 ann_token 来获取速度
            velocity = nusc.get('sample_annotation', ann['token'])['velocity']

            # nuScenes v1.0 schema for velocity is [vx, vy], but older devkit versions might have it nested.
            # We assume velocity is a list [vx, vy] or similar. Let's handle potential None.
            if velocity is None:
                velocity = [0.0, 0.0]  # 如果没有速度信息，则默认为0

            # 使用Box类来方便地获取center和heading
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

            cache['agents'][inst_token]['track'].append({
                'timestamp': sample['timestamp'],
                'ann_token': ann['token'],  # 存储annotation_token，非常重要！
                'pos': box.center[:2],
                'vel': np.array(velocity[:2]),
                'speed': np.linalg.norm(velocity[:2]),
                'heading': box.orientation.yaw_pitch_roll[0],
            })

        if not sample['next']:
            break
        sample = nusc.get('sample', sample['next'])

    # 对每个轨迹按时间戳排序，确保顺序正确
    for inst_token in cache['agents']:
        cache['agents'][inst_token]['track'].sort(key=lambda x: x['timestamp'])

    return cache



def calculate_acceleration_jerk(track_segment):
    """从位置轨迹段计算加速度和Jerk"""
    if len(track_segment) < 4: return None, None

    # 假设时间间隔是恒定的 (nuScenes ~0.5s)
    dt = (track_segment[1]['timestamp'] - track_segment[0]['timestamp']) / 1e6

    # 计算速度 (v_i = (p_{i+1} - p_i) / dt)
    velocities = [(p1['pos'] - p0['pos']) / dt for p0, p1 in zip(track_segment, track_segment[1:])]
    if len(velocities) < 3: return None, None

    # 计算加速度 (a_i = (v_{i+1} - v_i) / dt)
    accelerations = [(v1 - v0) / dt for v0, v1 in zip(velocities, velocities[1:])]
    if len(accelerations) < 2: return None, None

    # 计算Jerk (j_i = (a_{i+1} - a_i) / dt)
    jerks = [(a1 - a0) / dt for a0, a1 in zip(accelerations, accelerations[1:])]

    # 我们关心事件发生时刻的加速度和Jerk
    # track_segment的中心点对应事件时刻
    center_idx = len(velocities) // 2
    accel_magnitude = np.linalg.norm(accelerations[center_idx])
    jerk_magnitude = np.linalg.norm(jerks[center_idx - 1])

    return accel_magnitude, jerk_magnitude


def calculate_ttc(agent1_state, agent2_state):
    """计算两个agent之间的碰撞时间 (TTC)"""
    rel_pos = agent2_state['pos'] - agent1_state['pos']
    rel_vel = agent1_state['vel'] - agent2_state['vel']

    rel_speed = np.linalg.norm(rel_vel)
    if rel_speed < 1e-6:  # 相对速度几乎为0
        return float('inf')

    # 只有当agent1在追逐agent2时，TTC才有意义
    # 即相对速度方向与相对位置方向大致相反 (点积为负)
    if np.dot(rel_pos, rel_vel) > 0:
        return float('inf')

    ttc = np.linalg.norm(rel_pos) / rel_speed
    return ttc


def main():
    """主执行函数，协调整个分析流程。"""
    cfg = Config()

    # --- 1. 加载模型和数据 ---
    print("Loading nuScenes and LSTM model...")
    nusc = NuScenes(version=cfg.NUSCENES_VERSION, dataroot=cfg.NUSCENES_DATA_ROOT, verbose=False)

    # 假设模型已经能处理多类别输入 (例如方案B: Conditional LSTM)
    # num_agent_types = 3 # vehicle, pedestrian, bicycle
    # model = TrajectoryLSTM_Conditional(cfg, num_agent_types)
    model = TrajectoryLSTM(config=cfg)  # 使用简单模型作为示例

    try:
        model.load_state_dict(torch.load(cfg.MODEL_PATH))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file '{cfg.MODEL_PATH}' not found. Please ensure the trained model exists.")
        return

    all_final_events = []

    # --- 2. 遍历所有场景 ---
    for scene in tqdm(nusc.scene, desc="Processing Scenes"):

        # 2a. 为当前场景构建包含所有agent轨迹的缓存
        scene_cache = build_scene_track_cache(nusc, scene)

        # 2b. 对缓存中的所有轨迹进行离线预测
        all_predictions_in_scene = {}
        seq_len = cfg.HIST_LEN + cfg.PRED_LEN

        for inst_token, agent_data in scene_cache['agents'].items():
            track = agent_data['track']
            if len(track) < seq_len:
                continue

            all_predictions_in_scene[inst_token] = []
            for i in range(len(track) - seq_len + 1):
                hist_data = track[i: i + cfg.HIST_LEN]
                future_data = track[i + cfg.HIST_LEN: i + seq_len]

                origin = hist_data[-1]['pos']
                hist_rel = np.array([p['pos'] - origin for p in hist_data])

                with torch.no_grad():
                    hist_tensor = torch.tensor(hist_rel, dtype=torch.float32).unsqueeze(0)
                    pred_rel = model(hist_tensor).squeeze(0).numpy()

                pred_global = pred_rel + origin
                gt_global = np.array([p['pos'] for p in future_data])

                all_predictions_in_scene[inst_token].append({
                    'timestamp': hist_data[-1]['timestamp'],
                    'predicted': pred_global,
                    'ground_truth': gt_global
                })

        # 2c. 基于预测误差触发候选事件
        candidate_events = []
        for inst_token, agent_preds in all_predictions_in_scene.items():
            primary_agent_track = scene_cache['agents'].get(inst_token, {}).get('track', [])
            if not primary_agent_track:
                continue

            for pred_info in agent_preds:
                fde = np.linalg.norm(pred_info['ground_truth'][-1] - pred_info['predicted'][-1])

                state_at_pred_start = next((s for s in primary_agent_track if s['timestamp'] == pred_info['timestamp']),
                                           None)
                if not state_at_pred_start:
                    continue
                speed = state_at_pred_start['speed']

                if fde > cfg.FDE_THRESHOLD_ABS and fde > speed * cfg.FDE_THRESHOLD_REL_FACTOR:
                    candidate_events.append({
                        'instance_token': inst_token,
                        'timestamp': pred_info['timestamp'],
                    })

        # 2d. 对每个候选事件进行深度分析
        for candidate in tqdm(candidate_events, desc=f"Analyzing events in scene {scene['name']}", leave=False):
            analyzer = EventAnalyzer(
                config=cfg,
                scene_cache=scene_cache,
                primary_agent_token=candidate['instance_token'],
                event_timestamp=candidate['timestamp'],
                all_predictions=all_predictions_in_scene
            )
            event_details = analyzer.analyze()

            if event_details:
                all_final_events.append(event_details)

    # --- 3. 保存所有发现的事件到JSON文件 ---
    print(f"\nAnalysis complete. Found {len(all_final_events)} significant social interaction events.")
    if all_final_events:
        with open(cfg.OUTPUT_JSON_PATH, 'w') as f:
            json.dump(all_final_events, f, indent=4)
        print(f"Results saved to {cfg.OUTPUT_JSON_PATH}")
        print("\n--- Sample of Detected Events ---")
        print(json.dumps(all_final_events[0], indent=2))
    else:
        print("No significant events were found with the current thresholds.")


if __name__ == '__main__':
    main()