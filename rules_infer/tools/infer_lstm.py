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
    MODEL_SAVE_PATH = 'trajectory_lstm.pth'

    # --- 事件检测与分析参数 ---
    FDE_THRESHOLD_M = 2.0  # 最终位移误差的绝对阈值（米）
    FDE_VEL_MULTIPLIER = 1.5  # FDE的相对阈值，FDE > 速度 * 这个乘数
    TTC_THRESHOLD_S = 4.0  # 触发交互分析的碰撞时间阈值（秒）


class EventAnalyzer:
    def __init__(self, nusc, config, scene_cache, primary_agent_token, event_timestamp, all_predictions):
        self.nusc = nusc
        self.config = config
        self.scene_cache = scene_cache
        self.primary_agent_token = primary_agent_token
        self.event_timestamp = event_timestamp
        self.all_predictions = all_predictions

        self.primary_track = self.scene_cache['agents'].get(primary_agent_token)
        self.primary_state_at_event = self._get_state_from_track(self.primary_track, self.event_timestamp)

    def analyze(self):
        if not self.primary_state_at_event:
            return None

        # --- L2: 定性指标分析 (自身运动学) ---
        kinematics = self._calculate_kinematics()

        # --- L3: 关系指标分析 (与他车交互) ---
        interacting_agents, relational_metrics = self._analyze_relations()

        # --- 综合判断与打标签 ---
        event_label = self._determine_event_label(kinematics, relational_metrics)

        # 如果只是高误差但没有明显的交互或剧烈运动，可以过滤掉
        if event_label == "High_FDE_Only":
            return None

        # --- 整理并返回结果 ---
        event_data = {
            "scene_token": self.scene_cache['scene_token'],
            "start_timestamp": self.event_timestamp,
            "end_timestamp": self.event_timestamp + int(self.config.PRED_LEN * self.config.SAMPLE_INTERVAL_S * 1e6),
            "primary_agent_id": self.primary_agent_token,
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
        if not track: return None
        for state in track:
            if state['timestamp'] == timestamp:
                return state
        return None

    def _calculate_kinematics(self):
        """计算加速度、Jerk和偏航率"""
        kinematics = {}

        # 找到事件点在轨迹中的索引
        try:
            event_idx = next(
                i for i, state in enumerate(self.primary_track) if state['timestamp'] == self.event_timestamp)
        except StopIteration:
            return kinematics

        # 提取计算窗口
        win = self.config.KINEMATICS_WINDOW_SIZE
        if event_idx < win // 2 or event_idx >= len(self.primary_track) - (win // 2):
            return kinematics

        segment = self.primary_track[event_idx - win // 2: event_idx + win // 2 + 1]

        # 计算速度、加速度、Jerk (向量)
        pos = np.array([s['pos'] for s in segment])
        ts = np.array([s['timestamp'] for s in segment]) / 1e6
        dt = np.diff(ts)

        vel = np.diff(pos, axis=0) / dt[:, np.newaxis]
        accel = np.diff(vel, axis=0) / dt[1:, np.newaxis]
        jerk = np.diff(accel, axis=0) / dt[2:, np.newaxis]

        # 计算偏航率
        headings = np.unwrap([s['heading'] for s in segment])
        yaw_rate = np.diff(headings) / dt

        # 提取事件中心点的值
        center_idx = win // 2

        # 分解纵向和侧向加速度
        heading_vec = np.array([math.cos(segment[center_idx]['heading']), math.sin(segment[center_idx]['heading'])])
        center_accel = accel[center_idx - 1]
        kinematics['long_accel'] = np.dot(center_accel, heading_vec)
        kinematics['lat_accel'] = np.cross(heading_vec, center_accel)
        kinematics['jerk'] = np.linalg.norm(jerk[center_idx - 2]) if len(jerk) > 0 else 0
        kinematics['yaw_rate'] = yaw_rate[center_idx - 1]

        return kinematics

    def _analyze_relations(self):
        """分析与其他agent的关系：TTC, THW, Zone of Influence"""
        interacting_agents = []
        relational_metrics = {}

        # 获取FDE
        for pred in self.all_predictions.get(self.primary_agent_token, []):
            if pred['timestamp'] == self.event_timestamp:
                relational_metrics['fde'] = np.linalg.norm(pred['ground_truth'][-1] - pred['predicted'][-1])
                break

        min_ttc, min_thw = float('inf'), float('inf')
        ttc_agent, thw_agent = None, None

        primary_pos = self.primary_state_at_event['pos']
        primary_vel = self.primary_state_at_event['vel']
        primary_heading = self.primary_state_at_event['heading']

        # 遍历场景中的所有其他agent
        for other_token, other_track in self.scene_cache['agents'].items():
            if other_token == self.primary_agent_token:
                continue

            other_state = self._get_state_from_track(other_track, self.event_timestamp)
            if not other_state:
                continue

            # 1. 检查是否在影响区域内
            rel_pos = other_state['pos'] - primary_pos
            # 旋转到主车坐标系
            q_inv = Quaternion(axis=[0, 0, 1], angle=-primary_heading)
            rel_pos_local = q_inv.rotate(np.array([rel_pos[0], rel_pos[1], 0]))[:2]

            is_in_zone = (
                    -self.config.INTERACTION_ZONE_BACKWARD < rel_pos_local[0] < self.config.INTERACTION_ZONE_FORWARD and
                    abs(rel_pos_local[1]) < self.config.INTERACTION_ZONE_LATERAL
            )
            if not is_in_zone:
                continue

            # 2. 计算TTC (只对前方车辆有意义)
            if np.dot(rel_pos, primary_vel) > 0:  # 目标在前方
                rel_vel = primary_vel - other_state['vel']
                rel_speed = np.linalg.norm(rel_vel)
                if rel_speed > 1e-6 and np.dot(rel_pos, rel_vel) < 0:  # 正在接近
                    ttc = np.linalg.norm(rel_pos) / rel_speed
                    if ttc < min_ttc:
                        min_ttc = ttc
                        ttc_agent = {'token': other_token, 'state': other_state, 'where': 'front'}

            # 3. 计算THW (只对前方车辆有意义)
            if rel_pos_local[0] > 0 and self.primary_state_at_event['speed'] > 1.0:
                dist = np.linalg.norm(rel_pos)
                thw = dist / self.primary_state_at_event['speed']
                if thw < min_thw:
                    min_thw = thw
                    thw_agent = {'token': other_token, 'state': other_state}

        # 记录关键关系指标
        if min_ttc < self.config.TTC_THRESHOLD:
            relational_metrics['min_ttc'] = min_ttc
            if ttc_agent not in interacting_agents: interacting_agents.append(ttc_agent)
        if min_thw < self.config.THW_THRESHOLD_FOLLOW:
            relational_metrics['min_thw'] = min_thw
            if thw_agent and thw_agent not in interacting_agents: interacting_agents.append(thw_agent)

        # 4. 轨迹相交预测 (简化版)
        # TODO: A more complex implementation can be added here
        # This requires getting the other agent's prediction and checking for geometric intersection.
        # For now, we rely on TTC as a proxy for future collision.

        return interacting_agents, relational_metrics

    def _determine_event_label(self, kinematics, relational_metrics):
        """基于所有指标，给事件打上一个描述性标签"""
        jerk = kinematics.get('jerk', 0)
        long_accel = kinematics.get('long_accel', 0)
        yaw_rate = abs(kinematics.get('yaw_rate', 0))
        min_ttc = relational_metrics.get('min_ttc', float('inf'))

        is_hard_brake = long_accel < -3.0  # -3 m/s^2 是一个比较明显的刹车
        is_sudden_jerk = jerk > self.config.JERK_THRESHOLD
        is_sudden_swerve = yaw_rate > self.config.YAW_RATE_THRESHOLD
        is_ttc_critical = min_ttc < self.config.TTC_THRESHOLD

        if is_sudden_jerk or is_hard_brake:
            if is_ttc_critical:
                return "CRITICAL_BRAKE_FOR_LEAD_AGENT"
            else:
                return "SUDDEN_BRAKE_UNEXPLAINED"

        if is_sudden_swerve:
            if is_ttc_critical:
                return "EVASIVE_SWERVE_FOR_AGENT"
            else:
                return "ABRUPT_LANE_CHANGE_OR_SWERVE"

        # 针对低速下的“礼让”场景 (预测会走但没走)
        if self.primary_state_at_event['speed'] < 1.0 and relational_metrics.get('fde', 0) > 1.5:
            # 如果有其他车在交叉路径上，则可能是礼让
            # 此处需要更复杂的交叉口逻辑，简化为：
            if is_ttc_critical:  # 使用TTC作为交叉冲突的代理
                return "YIELDING_AT_INTERSECTION_OR_MERGE"

        return "High_FDE_Only"  # 默认，如果无明显特征则过滤


def build_scene_track_cache(nusc, scene):
    """为单个场景构建所有agent的轨迹缓存"""
    cache = {'scene_token': scene['token'], 'agents': {}}
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    while sample:
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            # 考虑车辆、行人、自行车
            if not any(cat in ann['category_name'] for cat in ['vehicle', 'human.pedestrian', 'bicycle']):
                continue

            inst_token = ann['instance_token']
            if inst_token not in cache['agents']:
                cache['agents'][inst_token] = []

            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            velocity = nusc.box_velocity(ann['token'])

            cache['agents'][inst_token].append({
                'timestamp': sample['timestamp'],
                'pos': box.center[:2],
                'vel': velocity[:2],
                'speed': np.linalg.norm(velocity[:2]),
                'heading': box.orientation.yaw_pitch_roll[0],
                'category': ann['category_name']
            })
        if not sample['next']: break
        sample = nusc.get('sample', sample['next'])
    return cache


def main():
    cfg = Config()

    # --- 1. 加载模型和数据 ---
    print("Loading nuScenes and LSTM model...")
    nusc = NuScenes(version=cfg.NUSCENES_VERSION, dataroot=cfg.NUSCENES_DATA_ROOT, verbose=False)
    model = TrajectoryLSTM(hist_len=cfg.HIST_LEN, pred_len=cfg.PRED_LEN)
    try:
        model.load_state_dict(torch.load(cfg.MODEL_PATH))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file '{cfg.MODEL_PATH}' not found. Please ensure the trained model exists.")
        return

    all_final_events = []

    # --- 2. 遍历场景 ---
    for scene in tqdm(nusc.scene, desc="Processing Scenes"):
        # --- 2a. 构建场景轨迹缓存 ---
        scene_cache = build_scene_track_cache(nusc, scene)

        # --- 2b. 离线预测 ---
        all_predictions_in_scene = {}
        seq_len = cfg.HIST_LEN + cfg.PRED_LEN

        for inst_token, track in scene_cache['agents'].items():
            if len(track) < seq_len: continue

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

        # --- 2c. 事件触发 ---
        candidate_events = []
        for inst_token, agent_preds in all_predictions_in_scene.items():
            primary_agent_track = scene_cache['agents'].get(inst_token)
            if not primary_agent_track: continue

            for pred_info in agent_preds:
                fde = np.linalg.norm(pred_info['ground_truth'][-1] - pred_info['predicted'][-1])

                state_at_pred_start = next((s for s in primary_agent_track if s['timestamp'] == pred_info['timestamp']),
                                           None)
                if not state_at_pred_start: continue
                speed = state_at_pred_start['speed']

                if fde > cfg.FDE_THRESHOLD_ABS and fde > speed * cfg.FDE_THRESHOLD_REL_FACTOR:
                    candidate_events.append({
                        'instance_token': inst_token,
                        'timestamp': pred_info['timestamp'],
                    })

        # --- 2d. 事件分析 ---
        for candidate in tqdm(candidate_events, desc=f"Analyzing events in scene {scene['name']}", leave=False):
            analyzer = EventAnalyzer(
                nusc=nusc,
                config=cfg,
                scene_cache=scene_cache,
                primary_agent_token=candidate['instance_token'],
                event_timestamp=candidate['timestamp'],
                all_predictions=all_predictions_in_scene
            )
            event_details = analyzer.analyze()

            if event_details:
                all_final_events.append(event_details)

    # --- 3. 保存结果 ---
    print(f"\nAnalysis complete. Found {len(all_final_events)} significant social interaction events.")
    with open(cfg.OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_final_events, f, indent=4)
    print(f"Results saved to {cfg.OUTPUT_JSON_PATH}")

    # 打印一些示例结果
    if all_final_events:
        print("\n--- Sample of Detected Events ---")
        for event in all_final_events[:3]:
            print(json.dumps(event, indent=2))
            print("-" * 20)


def get_agent_state_at_timestamp(nusc, scene, instance_token, target_timestamp):
    """获取指定agent在最接近目标时间戳时的状态"""
    sample_token = scene['first_sample_token']
    best_ann = None
    min_time_diff = float('inf')

    sample = nusc.get('sample', sample_token)
    while sample:
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            if ann['instance_token'] == instance_token:
                time_diff = abs(sample['timestamp'] - target_timestamp)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_ann = ann

        if not sample['next']:
            break
        sample = nusc.get('sample', sample['next'])

    if best_ann:
        box = Box(best_ann['translation'], best_ann['size'], Quaternion(best_ann['rotation']))
        velocity = nusc.box_velocity(box.token)[:2]
        return {
            'pos': box.center[:2],
            'vel': velocity,
            'heading': box.orientation.yaw_pitch_roll[0],
            'size': box.wlh[:2]
        }
    return None


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


def main_analysis_pipeline(config, nusc):
    """完整的分析流程"""
    print("\n--- Starting Analysis Pipeline ---")

    # --- Part A: 离线预测 ---
    print("Part A: Running offline predictions...")
    model = TrajectoryLSTM(config)
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}. Please train the model first.")
        return
    model.eval()

    all_predictions = {}  # 存储所有预测结果
    seq_len = config.HIST_LEN + config.PRED_LEN

    for scene in tqdm(nusc.scene, desc="A: Predicting trajectories per scene"):
        all_predictions[scene['token']] = {}
        instance_tracks = {}  # {instance_token: [pos_data, ...]}

        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', sample_token)
        while sample:
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                if 'vehicle' in ann['category_name']:
                    inst_token = ann['instance_token']
                    if inst_token not in instance_tracks:
                        instance_tracks[inst_token] = []

                    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                    instance_tracks[inst_token].append({
                        'pos': box.center[:2],
                        'timestamp': sample['timestamp'],
                        'sample_token': sample['token'],
                        'ann_token': ann['token']
                    })
            if not sample['next']: break
            sample = nusc.get('sample', sample['next'])

        # 对每个轨迹进行预测
        for inst_token, track in instance_tracks.items():
            if len(track) >= seq_len:
                all_predictions[scene['token']][inst_token] = []
                for i in range(len(track) - seq_len + 1):
                    hist_data = track[i: i + config.HIST_LEN]
                    future_data = track[i + config.HIST_LEN: i + seq_len]

                    origin = hist_data[-1]['pos']
                    hist_rel = np.array([p['pos'] - origin for p in hist_data])

                    with torch.no_grad():
                        hist_tensor = torch.tensor(hist_rel, dtype=torch.float32).unsqueeze(0)
                        pred_rel = model(hist_tensor).squeeze(0).numpy()

                    # 转回全局坐标
                    pred_global = pred_rel + origin
                    gt_global = np.array([p['pos'] for p in future_data])

                    all_predictions[scene['token']][inst_token].append({
                        'start_timestamp': hist_data[-1]['timestamp'],
                        'predicted': pred_global,
                        'ground_truth': gt_global
                    })

    # --- Part B: 事件触发 ---
    print("Part B: Triggering potential events...")
    candidate_events = []
    for scene_token, scene_preds in tqdm(all_predictions.items(), desc="B: Finding high-error events"):
        scene_info = nusc.get('scene', scene_token)
        for inst_token, agent_preds in scene_preds.items():
            for pred_info in agent_preds:
                gt = pred_info['ground_truth']
                pred = pred_info['predicted']

                # 计算FDE (Final Displacement Error)
                fde = np.linalg.norm(gt[-1] - pred[-1])

                # 使用动态阈值判断
                # 1. 获取当前速度
                primary_agent_state = get_agent_state_at_timestamp(nusc, scene_info, inst_token,
                                                                   pred_info['start_timestamp'])
                if primary_agent_state is None: continue
                speed = np.linalg.norm(primary_agent_state['vel'])

                # 2. 判断是否触发
                is_triggered = (fde > config.FDE_THRESHOLD_M and fde > speed * config.FDE_VEL_MULTIPLIER)

                if is_triggered:
                    candidate_events.append({
                        'scene_token': scene_token,
                        'instance_token': inst_token,
                        'timestamp': pred_info['start_timestamp'],
                        'fde': fde,
                        'speed_at_event': speed
                    })

    print(f"Found {len(candidate_events)} candidate events.")

    # --- Part C & D: 事件分析与归因, 并记录 ---
    print("Part C & D: Analyzing and logging events...")
    final_events = []
    for event in tqdm(candidate_events, desc="C&D: Analyzing events"):
        scene_token = event['scene_token']
        primary_agent_token = event['instance_token']
        event_timestamp = event['timestamp']

        scene_info = nusc.get('scene', scene_token)
        primary_agent_state = get_agent_state_at_timestamp(nusc, scene_info, primary_agent_token, event_timestamp)
        if not primary_agent_state: continue

        # --- L2 Metrics: 自我分析 ---
        # 提取事件前后的一小段轨迹来计算Jerk
        track = instance_tracks[primary_agent_token]  # instance_tracks from Part A
        event_idx = -1
        for i, p in enumerate(track):
            if p['timestamp'] == event_timestamp:
                event_idx = i
                break

        max_accel = max_jerk = None
        if event_idx != -1 and event_idx > 1 and event_idx < len(track) - 2:
            track_segment = track[event_idx - 2: event_idx + 3]  # 5个点
            max_accel, max_jerk = calculate_acceleration_jerk(track_segment)

        # --- L3 Metrics: 关系分析 ---
        min_ttc = float('inf')
        interacting_agent_token = None

        # 找到事件发生时场景中的所有其他agent
        sample = nusc.get('sample', nusc.get('sample_annotation',
                                             nusc.get_instance(primary_agent_token)['first_annotation_token'])[
            'sample_token'])
        # A simpler way to get the sample at the event time is needed. Let's find it.
        event_sample_token = None
        for p in track:
            if p['timestamp'] == event_timestamp:
                event_sample_token = p['sample_token']
                break

        if event_sample_token:
            sample = nusc.get('sample', event_sample_token)
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                if ann['instance_token'] != primary_agent_token and 'vehicle' in ann['category_name']:
                    other_agent_state = get_agent_state_at_timestamp(nusc, scene_info, ann['instance_token'],
                                                                     event_timestamp)
                    if not other_agent_state: continue

                    ttc = calculate_ttc(primary_agent_state, other_agent_state)
                    if ttc < min_ttc:
                        min_ttc = ttc
                        interacting_agent_token = ann['instance_token']

        # --- 事件标注 ---
        event_label = "High_FDE_Event"  # 默认标签
        if max_jerk and max_jerk > 10:  # 10 m/s^3 是一个比较大的值
            event_label = "Sudden_Maneuver"
            if min_ttc < config.TTC_THRESHOLD_S:
                # 检查交互对象是在前方还是侧方
                rel_pos = get_agent_state_at_timestamp(nusc, scene_info, interacting_agent_token, event_timestamp)[
                              'pos'] - primary_agent_state['pos']
                # 旋转到主车坐标系
                heading_quat = Quaternion(axis=[0, 0, 1], angle=primary_agent_state['heading'])
                rel_pos_local = heading_quat.inverse.rotate(np.array([rel_pos[0], rel_pos[1], 0]))

                if rel_pos_local[0] > 0:  # 交互对象在前方
                    event_label = "Hard_Brake_for_Lead_Vehicle"
                else:
                    event_label = "Evasive_Action_for_Side_Vehicle"

        final_events.append({
            'scene_token': scene_token,
            'primary_agent': primary_agent_token,
            'timestamp': event_timestamp,
            'event_label': event_label,
            'fde_m': event['fde'],
            'speed_kmh': event['speed_at_event'] * 3.6,
            'max_accel_ms2': max_accel,
            'max_jerk_ms3': max_jerk,
            'min_ttc_s': min_ttc if min_ttc != float('inf') else -1,
            'interacting_agent': interacting_agent_token
        })

    # --- 保存结果 ---
    if final_events:
        df = pd.DataFrame(final_events)
        output_path = 'social_interaction_events.csv'
        df.to_csv(output_path, index=False)
        print(f"\nAnalysis complete. Found {len(df)} significant events. Results saved to {output_path}")
        print("--- Event Log Preview ---")
        print(df.head())
    else:
        print("\nAnalysis complete. No significant events were found with the current thresholds.")


if __name__ == '__main__':
    cfg = Config()
    nusc = NuScenes(version=cfg.NUSCENES_VERSION, dataroot=cfg.NUSCENES_DATA_ROOT, verbose=False)
    main_analysis_pipeline(cfg, nusc)