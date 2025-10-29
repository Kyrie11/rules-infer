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

CONFIG = {
    # --- 数据和模型路径 ---
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <--- !!! 修改这里 !!!
    'version': 'v1.0-trainval',  # 建议先用 'v1.0-mini' 测试，然后换成 'v1.0-trainval'
    'model_path': 'nuscenes-lstm-model.pt',  # 你保存的模型权重文件
    'output_dir': 'eval_results',  # 保存可视化结果的文件夹

    # --- 模型和数据参数 (必须与训练时一致) ---
    'history_len': 8,
    'future_len': 12,
    'input_dim': 4,  # (x, y, is_near_tl, dist_to_tl) - 如果训练时没用地图，这里是 2
    'hidden_dim': 64,
    'output_dim': 2,
    'n_layers': 2,

    # --- 评估参数 ---
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # --- 地图参数 (如果训练时使用了) ---
    'traffic_light_distance_threshold': 30.0,
    # 定义FDE误差超过多少米被认为是“关键事件”
    'critical_event_threshold_fde': 2.0,  # 比如最终点误差超过2米

    # 定义FDE误差从一帧到下一帧的增量超过多少被认为是“关键事件”
    'critical_event_threshold_spike': 1.5,  # 例如，FDE在0.2秒内（相邻帧）增加了1.5米以上

    # 在识别出的事件窗口前后额外扩展多少帧作为上下文
    'critical_event_context_frames': 10,  # 往前和往后各扩展 10 帧 (即 5s)

    # --- 交互分析参数 ---
    'interaction_proximity_threshold': 25.0,  # 交互距离阈值（米）
    'interaction_heading_threshold_deg': 90.0, # 前方视角阈值（度）

    # 保存最终索引文件的路径
    'critical_event_index_file': 'critical_events.json'
}
# --- 配置参数 ---
NUSCENES_PATH = '/data0/senzeyu2/dataset/nuscenes'  # 修改为你的nuscenes数据集路径
NUSCENES_VERSION = 'v1.0-trainval'  # 或 'v1.0-trainval'
MODEL_PATH = 'trajectory_lstm.pth'
OUTPUT_JSON_PATH = 'social_events_with_humans.json'  # 新的输出文件名

# 事件检测参数
HIST_LEN = 8
PRED_LEN = 12
FDE_THRESHOLD = 2.0  # 对于行人，这个阈值可能需要调整
INTERACTION_RADIUS = 30.0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载模型
    model = TrajectoryLSTM(pred_len=PRED_LEN).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. 加载NuScenes
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_PATH, verbose=True)

    if NUSCENES_VERSION == 'v1.0-mini':
        eval_split_name = 'mini_val'
    else:  # 'v1.0-trainval'
        eval_split_name = 'val'
    scenes_for_split = create_splits_scenes()[eval_split_name]
    # 3. 遍历场景检测事件
    all_events = []
    scenes_to_process = [s for s in nusc.scene if s['name'] in scenes_for_split]
    for scene in tqdm(scenes_to_process, desc="Processing Scenes"):
        instance_trajs = get_trajectories_for_scene(nusc, scene)

        for main_agent_token, traj_data in instance_trajs.items():
            is_in_event = False
            current_event = {}
            sorted_frames = sorted(traj_data.keys())

            if len(sorted_frames) < HIST_LEN + PRED_LEN:
                continue

            for i in range(len(sorted_frames) - (HIST_LEN + PRED_LEN) + 1):
                frame_idx = sorted_frames[i + HIST_LEN - 1]
                history_frames = sorted_frames[i: i + HIST_LEN]
                future_gt_frames = sorted_frames[i + HIST_LEN: i + HIST_LEN + PRED_LEN]
                history_pos = np.array([traj_data[f]['pos'] for f in history_frames], dtype=np.float32)
                future_gt_pos = np.array([traj_data[f]['pos'] for f in future_gt_frames], dtype=np.float32)

                origin = history_pos[-1].copy()
                history_norm = torch.from_numpy(history_pos - origin).unsqueeze(0).to(device)

                with torch.no_grad():
                    future_pred_norm = model(history_norm)
                future_pred_pos = future_pred_norm.squeeze(0).cpu().numpy() + origin

                fde = np.linalg.norm(future_pred_pos[-1] - future_gt_pos[-1])

                if fde > FDE_THRESHOLD and not is_in_event:
                    is_in_event = True
                    current_event = {
                        "scene_token": scene['token'],
                        "main_agent_token": main_agent_token,
                        "start_frame": frame_idx
                    }
                elif fde <= FDE_THRESHOLD and is_in_event:
                    is_in_event = False
                    current_event["end_frame"] = frame_idx - 1
                    start_sample_token = traj_data[current_event['start_frame']]['sample_token']
                    interacting_agents = find_interacting_agents(
                        nusc, start_sample_token, main_agent_token,
                        traj_data[current_event['start_frame']]['pos'],
                        INTERACTION_RADIUS
                    )
                    current_event["interacting_agent_tokens"] = interacting_agents
                    all_events.append(current_event)

            if is_in_event:
                current_event["end_frame"] = sorted_frames[-1]
                start_sample_token = traj_data[current_event['start_frame']]['sample_token']
                interacting_agents = find_interacting_agents(
                    nusc, start_sample_token, main_agent_token,
                    traj_data[current_event['start_frame']]['pos'],
                    INTERACTION_RADIUS
                )
                current_event["interacting_agent_tokens"] = interacting_agents
                all_events.append(current_event)

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_events, f, indent=4)
    print(f"Detected {len(all_events)} events. Saved to {OUTPUT_JSON_PATH}")


def get_trajectories_for_scene(nusc, scene):
    trajs = {}
    current_token = scene['first_sample_token']
    frame_idx = 0
    while current_token != "":
        sample = nusc.get('sample', current_token)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            # --- MODIFICATION START ---
            is_vehicle = ann['category_name'].startswith('vehicle')
            is_human = ann['category_name'].startswith('human')

            if is_vehicle or is_human:
                # --- MODIFICATION END ---
                inst_token = ann['instance_token']
                if inst_token not in trajs:
                    trajs[inst_token] = {}
                trajs[inst_token][frame_idx] = {
                    'pos': ann['translation'][:2],
                    'sample_token': current_token
                }
        current_token = sample['next']
        frame_idx += 1
    return trajs


def find_interacting_agents(nusc, sample_token, main_agent_token, main_agent_pos, radius):
    interacting_agents = []
    sample = nusc.get('sample', sample_token)
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        inst_token = ann['instance_token']

        if inst_token == main_agent_token:
            continue

        # --- MODIFICATION START ---
        is_vehicle = ann['category_name'].startswith('vehicle')
        is_human = ann['category_name'].startswith('human')

        if not (is_vehicle or is_human):
            continue
        # --- MODIFICATION END ---

        other_pos = ann['translation'][:2]
        distance = np.linalg.norm(np.array(main_agent_pos) - np.array(other_pos))

        if distance < radius:
            interacting_agents.append(inst_token)

    return interacting_agents


if __name__ == '__main__':
    main()