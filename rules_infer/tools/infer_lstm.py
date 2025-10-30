import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.metrics import ADE, FDE
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from rules_infer.tools.motion_lstm import *
import math
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
    'FDE': 4.0,  # 例如，设为FDE的95百分位数
    'ICE_PEAK': 5.0, # 例如，设为Max ICE的95百分位数
    'ICE_BASELINE': 1.0, # 用于确定事件起止，可以设为中位数或稍高
}

# 事件窗口参数
PADDING_FRAMES_BEFORE = int(2.5 * FPS) # 向前回溯 2.5 秒
PADDING_FRAMES_AFTER = int(1.0 * FPS)  # 向后延伸 1.0 秒

# 交互Agent筛选参数
INTERACTION_RADIUS = 30.0 # 米
TOP_K_INTERACTING = 2 # 选取交互分数最高的K个agent

def calculate_ice_signal(pred_traj, gt_traj, fps):
    # ... (从Part 1复制完全相同的函数) ...
    ide = np.linalg.norm(pred_traj - gt_traj, axis=1)
    pred_vel = (pred_traj[1:] - pred_traj[:-1]) * fps
    gt_vel = (gt_traj[1:] - gt_traj[:-1]) * fps
    pred_speed = np.linalg.norm(pred_vel, axis=1)
    gt_speed = np.linalg.norm(gt_vel, axis=1)
    ive = np.abs(pred_speed - gt_speed)
    ive = np.append(ive, ive[-1])
    pred_accel = (pred_vel[1:] - pred_vel[:-1]) * fps
    gt_accel = (gt_vel[1:] - gt_vel[:-1]) * fps
    pred_accel_scalar = np.linalg.norm(pred_accel, axis=1)
    gt_accel_scalar = np.linalg.norm(gt_accel, axis=1)
    iae = np.abs(pred_accel_scalar - gt_accel_scalar)
    iae = np.append(iae, [iae[-1], iae[-1]])
    w_pos = 1.0
    w_acc = 0.5
    ice = w_pos * ide + w_acc * iae
    return ice


def find_interacting_agents(nusc, key_agent_token, scene_token, event_onset_frame_idx):
    """在事件发生时刻，寻找潜在的交互Agent"""
    sample_token = nusc.get('sample', nusc.get('scene', scene_token)['first_sample_token'])[
                       'next'] * event_onset_frame_idx
    if event_onset_frame_idx == 0:
        sample_token = nusc.get('scene', scene_token)['first_sample_token']

    try:
        sample = nusc.get('sample', sample_token)
    except KeyError:
        return []

    key_agent_ann = nusc.get_sample_annotation(nusc.get('instance', key_agent_token)['first_annotation_token'],
                                               sample_token)
    key_agent_pos = np.array(key_agent_ann['translation'][:2])

    potential_interactors = []

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == key_agent_token or 'vehicle' not in ann['category_name']:
            continue

        interactor_pos = np.array(ann['translation'][:2])
        distance = np.linalg.norm(key_agent_pos - interactor_pos)

        if distance < INTERACTION_RADIUS:
            # 简单的交互分数：距离越近，分数越高
            interaction_score = 1 / (distance + 1e-6)
            potential_interactors.append({
                'token': ann['instance_token'],
                'score': interaction_score,
                'distance': distance
            })

    # 按分数排序并返回Top-K
    sorted_interactors = sorted(potential_interactors, key=lambda x: x['score'], reverse=True)
    return [agent['token'] for agent in sorted_interactors[:TOP_K_INTERACTING]]


def main():
    print(f"Using device: {DEVICE}")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_PATH, verbose=False)
    helper = PredictHelper(nusc)

    # 加载模型
    encoder = Encoder(input_dim=INPUT_DIM, emb_dim=32, hid_dim=64, n_layers=2, dropout=0.5)
    decoder = Decoder(output_dim=OUTPUT_DIM, emb_dim=32, hid_dim=64, n_layers=2, dropout=0.5)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_events = []

    val_scenes = [s for s in nusc.scene if nusc.get('log', s['log_token'])['logfile'].startswith('n008')]

    with torch.no_grad():
        for scene in tqdm(val_scenes, desc="Detecting Events in Scenes"):
            # 在一个场景中，我们可能需要对一个agent在不同时间点进行预测
            # 为了简化，我们只在每个agent轨迹的中间点进行一次预测
            # 更完整的实现会使用滑动窗口

            for instance_token in helper.get_annotations_for_scene(scene['name']):
                annotation = nusc.get('sample_annotation',
                                      nusc.get('instance', instance_token)['first_annotation_token'])
                if 'vehicle' not in annotation['category_name']:
                    continue

                # 选取一个采样点进行预测 (这里简化为场景的中间采样点)
                mid_sample_token = helper.get_sample_token_for_scene(scene['name'], 0.5)

                past_traj = helper.get_past_for_agent(instance_token, mid_sample_token, seconds=OBS_LEN / FPS,
                                                      in_agent_frame=False)
                future_traj = helper.get_future_for_agent(instance_token, mid_sample_token, seconds=PRED_LEN / FPS,
                                                          in_agent_frame=False)

                if past_traj.shape[0] < OBS_LEN or future_traj.shape[0] < PRED_LEN:
                    continue

                # 模型预测
                obs_traj_tensor = torch.tensor(past_traj, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                pred_future_traj_tensor = model(obs_traj_tensor, PRED_LEN)
                pred_future_traj = pred_future_traj_tensor.squeeze(0).cpu().numpy()

                # 计算误差
                fde = np.linalg.norm(pred_future_traj[-1] - future_traj[-1])
                ice_signal = calculate_ice_signal(pred_future_traj, future_traj, FPS)
                max_ice = np.max(ice_signal)

                # --- 核心判断逻辑 ---
                if fde > THRESHOLDS['FDE'] or max_ice > THRESHOLDS['ICE_PEAK']:
                    # 触发！这是一个潜在的社会性事件

                    # 1. 确定事件时间窗口
                    t_peak_relative = np.argmax(ice_signal)

                    # 查找开始和结束点
                    above_baseline = np.where(ice_signal > THRESHOLDS['ICE_BASELINE'])[0]
                    t_start_relative = above_baseline[0] if len(above_baseline) > 0 else t_peak_relative
                    t_end_relative = above_baseline[-1] if len(above_baseline) > 0 else t_peak_relative

                    # 转换为绝对帧号
                    current_sample = nusc.get('sample', mid_sample_token)
                    current_frame_idx = nusc.get('sample', scene['first_sample_token'])['next']
                    current_frame_idx_val = 0
                    while current_frame_idx != mid_sample_token:
                        current_frame_idx_val += 1
                        try:
                            current_frame_idx = nusc.get('sample', current_frame_idx)['next']
                        except KeyError:
                            break

                    peak_error_frame = current_frame_idx_val + t_peak_relative

                    event_start_frame = current_frame_idx_val + t_start_relative - PADDING_FRAMES_BEFORE
                    event_end_frame = current_frame_idx_val + t_end_relative + PADDING_FRAMES_AFTER

                    # 保证帧号不越界
                    event_start_frame = max(0, event_start_frame)
                    event_end_frame = min(scene['nbr_samples'] - 1, event_end_frame)

                    # 2. 识别交互Agent
                    # 我们在误差开始增大的那一刻寻找交互者
                    onset_frame_idx = current_frame_idx_val + t_start_relative
                    interacting_agent_tokens = find_interacting_agents(nusc, instance_token, scene['token'],
                                                                       onset_frame_idx)

                    # 3. 打包事件信息
                    event_data = {
                        "scene_token": scene['token'],
                        "scene_name": scene['name'],
                        "key_agent_token": instance_token,
                        "interacting_agent_tokens": interacting_agent_tokens,
                        "event_start_frame": event_start_frame,
                        "event_end_frame": event_end_frame,
                        "peak_error_frame": peak_error_frame,
                        "trigger_metrics": {
                            "FDE": float(fde),
                            "max_ICE": float(max_ice)
                        }
                    }
                    all_events.append(event_data)

    # 保存到JSON文件
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_events, f, indent=4)

    print(f"\nDetection complete. Found {len(all_events)} potential social events.")
    print(f"Results saved to {OUTPUT_JSON_PATH}")


if __name__ == '__main__':
    main()