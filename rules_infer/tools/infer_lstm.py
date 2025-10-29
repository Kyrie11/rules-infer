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

    # --- 模型与训练参数 ---
    HIST_LEN = 8  # 历史轨迹长度 (N_in)
    PRED_LEN = 12  # 预测轨迹长度 (N_out)
    INPUT_DIM = 2  # 输入特征维度 (x, y)
    OUTPUT_DIM = 2  # 输出特征维度 (x, y)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20  # 演示目的，实际可增加
    MODEL_SAVE_PATH = 'trajectory_lstm.pth'
    MIN_EVENT_DURATION_FRAMES = 3  # 一个事件必须持续至少3帧 (约1.5秒)才被认为是有效事件
    INTERACTION_LOOKBACK_SEC = 3.0


class EventAnalyzer:
    # __init__ 和其他方法保持不变，除了 _analyze_relations

    def __init__(self, config, scene_cache, primary_agent_token, event_info, all_predictions):
        self.config = config
        self.scene_cache = scene_cache
        self.primary_agent_token = primary_agent_token

        # event_info 现在是一个包含 start, end, peak_timestamp 的字典
        self.event_info = event_info
        self.event_peak_timestamp = event_info['peak_timestamp']

        self.all_predictions = all_predictions

        self.primary_data = self.scene_cache['agents'].get(primary_agent_token, {})
        self.primary_track = self.primary_data.get('track', [])
        # 分析的基准状态是事件峰值时刻的状态
        self.primary_state_at_event = self._get_state_from_track(self.primary_track, self.event_peak_timestamp)

    def analyze(self):
        """执行完整的事件分析并返回结构化的事件数据。"""
        if not self.primary_state_at_event:
            return None

        kinematics = self._calculate_kinematics(self.event_peak_timestamp)  # 传入峰值时间戳
        interacting_agents, relational_metrics = self._analyze_relations() # 使用新的回溯分析方法

        event_label = self._determine_event_label(kinematics, relational_metrics)
        if event_label == "High_FDE_Only":
            return None

        event_data = {
            "scene_token": self.scene_cache['scene_token'],
            # 使用事件区间的真实起止时间
            "start_timestamp": self.event_info['start_timestamp'],
            "end_timestamp": self.event_info['end_timestamp'],
            "duration_s": (self.event_info['end_timestamp'] - self.event_info['start_timestamp']) / 1e6,
            "primary_agent_id": self.primary_agent_token,
            "primary_agent_category": self.primary_data.get('category', 'unknown'),
            "interacting_agent_ids": [agent['token'] for agent in interacting_agents],
            "event_type_label": event_label,
            "metrics_snapshot": {
                "peak_fde": relational_metrics.get('peak_fde'),  # 改为 peak_fde
                # ... (其他指标) ...
            }
        }
        # 为了简洁，此处省略了完整的字典构建，其结构与之前类似
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
        """
        在事件发生前回溯一段时间，分析关系指标的演变，以找到真正的交互对象。
        """
        interacting_agents = []
        relational_metrics = {}

        # 1. 获取峰值FDE
        agent_predictions = self.all_predictions.get(self.primary_agent_token, [])
        pred_info = next((p for p in agent_predictions if p['timestamp'] == self.event_peak_timestamp), None)
        if pred_info:
            relational_metrics['peak_fde'] = np.linalg.norm(pred_info['ground_truth'][-1] - pred_info['predicted'][-1])

        # 2. 建立回溯时间窗口
        lookback_usec = int(self.config.INTERACTION_LOOKBACK_SEC * 1e6)
        analysis_start_time = self.event_peak_timestamp - lookback_usec

        # 提取主车在回溯窗口内的轨迹段
        primary_segment = [s for s in self.primary_track if analysis_start_time <= s['timestamp'] <= self.event_peak_timestamp]
        if not primary_segment:
            return [], relational_metrics

        # 3. 遍历所有其他agent，在回溯窗口内寻找最小TTC
        global_min_ttc = float('inf')
        causal_agent_token = None

        for other_token, other_data in self.scene_cache['agents'].items():
            if other_token == self.primary_agent_token:
                continue

            other_track = other_data.get('track', [])
            other_segment = [s for s in other_track if analysis_start_time <= s['timestamp'] <= self.event_peak_timestamp]
            if not other_segment:
                continue

            # 对齐两个agent在窗口内的轨迹
            # (这是一个简化版对齐，假设时间戳能匹配)
            for primary_state_hist in primary_segment:
                ts = primary_state_hist['timestamp']
                other_state_hist = self._get_state_from_track(other_segment, ts)
                if not other_state_hist:
                    continue

                # 计算历史时刻的TTC
                # (这里可以复用全局的 calculate_ttc 函数)
                current_ttc = calculate_ttc(primary_state_hist, other_state_hist)

                if current_ttc < global_min_ttc:
                    global_min_ttc = current_ttc
                    causal_agent_token = other_token

        # 4. 如果找到了因果agent，记录下来
        if global_min_ttc < self.config.TTC_THRESHOLD:
            relational_metrics['min_ttc_s'] = global_min_ttc
            interacting_agents.append({'token': causal_agent_token})
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
            velocity = nusc.get('sample_annotation', ann['token']).get('velocity', [0.0, 0.0])

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

class NumpyEncoder(json.JSONEncoder):
    """
    自定义编码器，用于处理json.dump无法序列化的NumPy类型。
    这是解决相关TypeError的标准方法。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)      # 将NumPy整数转换为Python整数
        elif isinstance(obj, np.floating):
            return float(obj)    # 将NumPy浮点数转换为Python浮点数
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 将NumPy数组转换为Python列表
        else:
            return super(NumpyEncoder, self).default(obj)

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
        high_fde_timestamps = {}  # key: inst_token, value: list of (timestamp, fde)

        for inst_token, agent_preds in all_predictions_in_scene.items():
            # >>> 问题3的解决方案：在这里过滤 primary agent 的类别 <<<
            agent_category = scene_cache['agents'].get(inst_token, {}).get('category', '')
            if 'vehicle' not in agent_category:
                continue

            primary_agent_track = scene_cache['agents'][inst_token]['track']

            for pred_info in agent_preds:
                fde = np.linalg.norm(pred_info['ground_truth'][-1] - pred_info['predicted'][-1])
                state = next((s for s in primary_agent_track if s['timestamp'] == pred_info['timestamp']), None)
                if not state: continue
                speed = state['speed']

                if fde > cfg.FDE_THRESHOLD_ABS and fde > speed * cfg.FDE_THRESHOLD_REL_FACTOR:
                    if inst_token not in high_fde_timestamps:
                        high_fde_timestamps[inst_token] = []
                    high_fde_timestamps[inst_token].append((pred_info['timestamp'], fde))

        # 步骤 2c-2: 对高FDE时间点进行分组，形成事件区间
        candidate_event_groups = []
        for inst_token, ts_fde_list in high_fde_timestamps.items():
            if not ts_fde_list: continue

            ts_fde_list.sort()  # 按时间戳排序
            current_group = [ts_fde_list[0]]

            for i in range(1, len(ts_fde_list)):
                # 如果当前时间戳与上一时间戳是连续的 (间隔约0.5s)
                time_diff = (ts_fde_list[i][0] - ts_fde_list[i - 1][0]) / 1e6
                if time_diff < cfg.SAMPLE_INTERVAL_S * 1.5:
                    current_group.append(ts_fde_list[i])
                else:
                    candidate_event_groups.append((inst_token, current_group))
                    current_group = [ts_fde_list[i]]
            candidate_event_groups.append((inst_token, current_group))

        # 步骤 2d: 对每个事件区间进行分析
        for inst_token, group in candidate_event_groups:
            # >>> 问题1&2的解决方案：过滤短事件 <<<
            if len(group) < cfg.MIN_EVENT_DURATION_FRAMES:
                continue

            # 找到峰值FDE的时刻作为分析的关键帧
            peak_timestamp, peak_fde = max(group, key=lambda item: item[1])

            event_info = {
                'start_timestamp': group[0][0],
                'end_timestamp': group[-1][0],
                'peak_timestamp': peak_timestamp,
                'peak_fde': peak_fde,
                'duration_frames': len(group)
            }

            analyzer = EventAnalyzer(
                config=cfg,
                scene_cache=scene_cache,
                primary_agent_token=inst_token,
                event_info=event_info,  # 传递整个事件信息
                all_predictions=all_predictions_in_scene
            )
            event_details = analyzer.analyze()  # analyze 内部现在使用 event_info

            if event_details:
                all_final_events.append(event_details)

    # --- 3. 保存所有发现的事件到JSON文件 ---
    print(f"\nAnalysis complete. Found {len(all_final_events)} significant social interaction events.")
    if all_final_events:
        with open(cfg.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_final_events, f, indent=4, cls=NumpyEncoder)
        print(f"Results saved to {cfg.OUTPUT_JSON_PATH}")
        print("\n--- Sample of Detected Events ---")
        print(json.dumps(all_final_events[0], indent=2))
    else:
        print("No significant events were found with the current thresholds.")


if __name__ == '__main__':
    main()