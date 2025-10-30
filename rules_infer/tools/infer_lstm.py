import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from rules_infer.tools.motion_lstm import *
from rules_infer.tools.config import Config
from rules_infer.tools.analyze_error_distribution import calculate_ice_signal, get_all_instances_in_scene
import json
from collections import defaultdict

NUSCENES_PATH = '/path/to/your/nuscenes'
NUSCENES_VERSION = 'v1.0-trainval'
MODEL_PATH = './lstm_model.pth'
OUTPUT_JSON_PATH = './social_events.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型和轨迹参数
OBS_LEN = 8
PRED_LEN = 12
INPUT_DIM = 2
OUTPUT_DIM = 2
FPS = 2

# !!! 关键阈值 !!!
# 请根据Part 1脚本的输出来设定这些值
THRESHOLDS = {
    'FDE': 5.0,  # 例如，设为FDE的95百分位数
    'ICE_PEAK': 7.0, # 例如，设为Max ICE的95百分位数
    'ICE_BASELINE': 1.0, # 用于确定事件起止，可以设为中位数或稍高
}

# 事件窗口参数
PADDING_FRAMES_BEFORE = int(2.5 * FPS) # 向前回溯 2.5 秒
PADDING_FRAMES_AFTER = int(1.0 * FPS)  # 向后延伸 1.0 秒

# 交互Agent筛选参数
INTERACTION_RADIUS = 30.0 # 米
TOP_K_INTERACTING = 2 # 选取交互分数最高的K个agent


def find_interacting_agents(nusc, key_agent_token, scene, onset_frame_idx, config):
    try:
        sample_token = get_sample_token_by_index(nusc, scene, onset_frame_idx)
        sample = nusc.get('sample', sample_token)
        key_agent_ann = nusc.get_sample_annotation_for_instance(key_agent_token, sample_token)
    except (KeyError, IndexError):
        return []
    key_agent_pos = np.array(key_agent_ann['translation'][:2])
    potential_interactors = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == key_agent_token or 'vehicle' not in ann['category_name']: continue
        interactor_pos = np.array(ann['translation'][:2])
        distance = np.linalg.norm(key_agent_pos - interactor_pos)
        if distance < config.INTERACTION_RADIUS_M:
            potential_interactors.append({'token': ann['instance_token'], 'score': 1 / (distance + 1e-6)})
    sorted_interactors = sorted(potential_interactors, key=lambda x: x['score'], reverse=True)
    return [agent['token'] for agent in sorted_interactors[:config.TOP_K_INTERACTING]]


def get_sample_token_by_index(nusc, scene, sample_idx):
    if sample_idx < 0 or sample_idx >= scene['nbr_samples']: raise IndexError("Sample index out of bounds.")
    current_token = scene['first_sample_token']
    for _ in range(int(sample_idx)):
        sample = nusc.get('sample', current_token)
        current_token = sample['next']
        if not current_token: break
    return current_token


def get_full_trajectory(nusc, helper, instance_token, scene):
    """获取一个instance在整个场景中的完整轨迹"""
    traj = []
    current_sample_token = scene['first_sample_token']
    while current_sample_token:
        try:
            ann = helper.get_sample_annotation(instance_token, current_sample_token)
            traj.append(ann['translation'][:2])
        except KeyError:
            # 在某些帧，该instance可能没有标注，我们选择填充None或直接跳过
            # 为了简单，我们只记录有标注的帧，但这可能导致轨迹不连续
            # 更复杂的实现会处理不连续性，但这里我们先简化
            pass
        current_sample_token = nusc.get('sample', current_sample_token)['next']
    return np.array(traj)


def merge_overlapping_events(events):
    """将时间上重叠的事件合并成一个"""
    if not events:
        return []

    # 按开始时间排序
    events.sort(key=lambda x: x['event_start_frame'])

    merged_events = [events[0]]

    for current_event in events[1:]:
        last_merged = merged_events[-1]

        # 检查是否重叠
        if current_event['event_start_frame'] <= last_merged['event_end_frame']:
            # 合并事件
            last_merged['event_end_frame'] = max(last_merged['event_end_frame'], current_event['event_end_frame'])
            # 合并触发指标（取最大值）
            last_merged['trigger_metrics']['FDE'] = max(last_merged['trigger_metrics']['FDE'],
                                                        current_event['trigger_metrics']['FDE'])
            last_merged['trigger_metrics']['max_ICE'] = max(last_merged['trigger_metrics']['max_ICE'],
                                                            current_event['trigger_metrics']['max_ICE'])
            # 合并交互者（去重）
            last_merged['interacting_agent_tokens'] = list(
                set(last_merged['interacting_agent_tokens'] + current_event['interacting_agent_tokens']))
        else:
            merged_events.append(current_event)

    return merged_events


def main():
    config = Config()
    # --- 在这里更新您的阈值 ---
    config.FDE_THRESHOLD = 5.0
    config.ICE_PEAK_THRESHOLD = 7.0
    config.ICE_BASELINE = 1.0
    # -------------------------

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    nusc = NuScenes(version=config.NUSCENES_VERSION, dataroot=config.NUSCENES_DATA_ROOT, verbose=False)
    helper = PredictHelper(nusc)

    model = TrajectoryLSTM(config).to(DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    final_events_by_scene = defaultdict(list)
    val_scenes = [s for s in nusc.scene if nusc.get('log', s['log_token'])['logfile'].startswith('n008')]

    with torch.no_grad():
        for scene in tqdm(val_scenes, desc="Detecting Events in Scenes"):
            all_instance_tokens_in_scene = get_all_instances_in_scene(nusc, scene)

            for instance_token in all_instance_tokens_in_scene:
                try:
                    first_ann_token = nusc.get('instance', instance_token)['first_annotation_token']
                    if 'vehicle' not in nusc.get('sample_annotation', first_ann_token)['category_name']: continue
                except KeyError:
                    continue

                # --- 滑动窗口逻辑 ---
                full_track_global = get_full_trajectory(nusc, helper, instance_token, scene)
                seq_len = config.HIST_LEN + config.PRED_LEN

                if len(full_track_global) < seq_len:
                    continue

                for i in range(len(full_track_global) - seq_len + 1):
                    current_frame_idx = i  # 这是相对于轨迹开始的帧索引

                    # 提取当前窗口的轨迹
                    past_traj_global = full_track_global[i: i + config.HIST_LEN]
                    future_traj_global = full_track_global[i + config.HIST_LEN: i + seq_len]

                    # 坐标系归一化
                    ref_point = past_traj_global[-1]
                    past_traj_relative = past_traj_global - ref_point
                    obs_tensor = torch.tensor(past_traj_relative, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                    # 模型预测并反归一化
                    pred_future_traj_relative = model(obs_tensor).squeeze(0).cpu().numpy()
                    pred_future_traj_global = pred_future_traj_relative + ref_point

                    # 计算误差
                    fde = np.linalg.norm(pred_future_traj_global[-1] - future_traj_global[-1])
                    ice_signal = calculate_ice_signal(pred_future_traj_global, future_traj_global, config.FPS)
                    max_ice = np.max(ice_signal)

                    # --- 核心判断逻辑 ---
                    if fde > config.FDE_THRESHOLD or max_ice > config.ICE_PEAK_THRESHOLD:
                        t_peak_relative = np.argmax(ice_signal)
                        above_baseline = np.where(ice_signal > config.ICE_BASELINE)[0]
                        t_start_relative = above_baseline[0] if len(above_baseline) > 0 else t_peak_relative
                        t_end_relative = above_baseline[-1] if len(above_baseline) > 0 else t_peak_relative

                        # 转换为场景绝对帧号
                        prediction_start_frame = current_frame_idx + config.HIST_LEN

                        peak_error_frame = prediction_start_frame + t_peak_relative
                        event_start_frame = prediction_start_frame + t_start_relative - config.PADDING_FRAMES_BEFORE
                        event_end_frame = prediction_start_frame + t_end_relative + config.PADDING_FRAMES_AFTER

                        onset_frame_idx = prediction_start_frame + t_start_relative
                        interacting_agents = find_interacting_agents(nusc, instance_token, scene, onset_frame_idx,
                                                                     config)

                        event_data = {
                            "key_agent_token": instance_token,
                            "interacting_agent_tokens": interacting_agents,
                            "event_start_frame": int(max(0, event_start_frame)),
                            "event_end_frame": int(min(scene['nbr_samples'] - 1, event_end_frame)),
                            "peak_error_frame": int(peak_error_frame),
                            "trigger_metrics": {"FDE": float(fde), "max_ICE": float(max_ice)}
                        }
                        final_events_by_scene[scene['token']].append(event_data)

    # --- 合并每个场景中重叠的事件 ---
    merged_final_events = []
    for scene_token, events in final_events_by_scene.items():
        # 对每个key_agent的事件分别进行合并
        events_by_agent = defaultdict(list)
        for e in events:
            events_by_agent[e['key_agent_token']].append(e)

        for agent_token, agent_events in events_by_agent.items():
            merged_agent_events = merge_overlapping_events(agent_events)
            for mev in merged_agent_events:
                mev['scene_token'] = scene_token
                mev['scene_name'] = nusc.get('scene', scene_token)['name']
                merged_final_events.append(mev)

    # 保存到JSON文件
    with open(config.EVENT_JSON_OUTPUT_PATH, 'w') as f:
        json.dump(merged_final_events, f, indent=4)

    print(f"\nDetection complete. Found {len(merged_final_events)} potential social events.")
    print(f"Results saved to {config.EVENT_JSON_OUTPUT_PATH}")


if __name__ == '__main__':
    main()