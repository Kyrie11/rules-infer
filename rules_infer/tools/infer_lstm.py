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
from rules_infer.dataset.nuscenes import NuScenesTrajectoryDataset
from rules_infer.tools.motion_lstm import *


class Config:
    # --- 路径与数据集 ---
    NUSCENES_DATA_ROOT = '/data0/senzeyu2/dataset/nuscenes'  # !!! 修改为你的路径 !!!
    NUSCENES_VERSION = 'v1.0-trainval'
    MODEL_PATH = 'trajectory_lstm.pth'
    OUTPUT_JSON_PATH = 'social_interaction_events.json'

    # --- 模型参数 ---
    HIST_LEN = 8
    PRED_LEN = 12

    # --- 事件触发阈值 (L1) ---
    FDE_THRESHOLD_ABS = 2.5  # 绝对FDE阈值 (米)
    FDE_THRESHOLD_REL_FACTOR = 1.5  # 相对FDE阈值: FDE > speed * factor

    # --- 事件分析阈值 (L2 & L3) ---
    JERK_THRESHOLD = 8.0  # 急动度阈值 (m/s^3)
    YAW_RATE_THRESHOLD = 0.4  # 偏航率阈值 (rad/s, 约23度/秒)
    TTC_THRESHOLD = 4.0  # 碰撞时间阈值 (秒)
    THW_THRESHOLD_FOLLOW = 2.5  # 跟车时距阈值 (秒)

    # --- 影响区域定义 (Zone of Influence) ---
    INTERACTION_ZONE_FORWARD = 50.0  # 前方 (米)
    INTERACTION_ZONE_BACKWARD = 10.0  # 后方 (米)
    INTERACTION_ZONE_LATERAL = 5.0  # 两侧 (米, 约一个半车道)

    # --- 时间窗口 ---
    SAMPLE_INTERVAL_S = 0.5  # nuScenes采样间隔约0.5秒
    KINEMATICS_WINDOW_SIZE = 5  # 用于计算Jerk的点数 (中心点+/-2)


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


if __name__ == '__main__':
    main()