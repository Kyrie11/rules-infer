import torch
from torch.utils.data import DataLoader, random_split
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
# 确保 nuscenes-devkit 和其他库已安装
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from collections import defaultdict
import numpy as np
from rules_infer.dataset.nuscenes import NuScenesTrajectoryDataset
from rules_infer.tools.motion_lstm import Encoder, Decoder, Seq2Seq

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


def find_interacting_agents(key_agent_token, scene_token, frame_idx, full_trajectories, all_agents_in_scene,
                            proximity_thresh, heading_thresh_deg):
    """
    在指定场景的某一帧中，寻找与关键agent有交互的其他agent。
    """
    interacting_agents = []

    # 1. 获取关键agent在当前帧的状态
    key_traj = full_trajectories.get((scene_token, key_agent_token))
    if key_traj is None or frame_idx >= len(key_traj):
        return []
    key_pos = key_traj[frame_idx]

    # 2. 近似计算关键agent的朝向向量
    key_heading_vec = None
    if frame_idx > 0 and len(key_traj) > frame_idx: # <-- 增加 len 检查
        prev_pos = key_traj[frame_idx - 1]
        # 确保agent在移动，否则没有朝向
        if np.linalg.norm(key_pos - prev_pos) > 0.1:
            key_heading_vec = key_pos - prev_pos

    # 3. 遍历场景中的所有其他agent
    for other_token in all_agents_in_scene:
        if other_token == key_agent_token:
            continue

        other_traj = full_trajectories.get((scene_token, other_token))
        if other_traj is None or frame_idx >= len(other_traj):
            continue

        other_pos = other_traj[frame_idx]

        # 4. 条件1：检查空间距离
        distance = np.linalg.norm(key_pos - other_pos)
        if distance >= proximity_thresh:
            continue  # 太远了，跳过

        # 5. 条件2：检查是否在关键agent的前方视野内 (可选但推荐)
        is_in_front = True
        if key_heading_vec is not None:
            vec_to_other = other_pos - key_pos
            # 防止静止的other agent导致零向量
            if np.linalg.norm(vec_to_other) > 1e-4:
                cos_angle = np.dot(key_heading_vec, vec_to_other) / (
                            np.linalg.norm(key_heading_vec) * np.linalg.norm(vec_to_other))
                angle_deg = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                if angle_deg > (heading_thresh_deg / 2):
                    is_in_front = False

        if is_in_front:
            interacting_agents.append(other_token)

    return interacting_agents

def calculate_ade(pred, gt):
    """计算平均位移误差 (ADE)"""
    # pred/gt shape: [batch_size, future_len, 2]
    error = torch.norm(pred - gt, p=2, dim=2)  # 计算每个时间点的欧氏距离
    ade = torch.mean(error, dim=1)  # 在时间维度上求平均
    return torch.mean(ade)  # 在 batch 维度上求平均


def calculate_fde(pred, gt):
    """计算最终位移误差 (FDE)"""
    # pred/gt shape: [batch_size, future_len, 2]
    error = torch.norm(pred[:, -1, :] - gt[:, -1, :], p=2, dim=1)  # 只计算最后一个时间点的欧氏距离
    return torch.mean(error)  # 在 batch 维度上求平均


def visualize_prediction(history, gt_future, pred_future, save_path):
    """可视化单个预测结果"""
    history = history.cpu().numpy()
    gt_future = gt_future.cpu().numpy()
    pred_future = pred_future.cpu().numpy()

    plt.figure(figsize=(8, 8))

    # 绘制历史轨迹 (蓝色)
    plt.plot(history[:, 0], history[:, 1], 'bo-', label='History')

    # 绘制真实未来轨迹 (绿色)
    plt.plot(gt_future[:, 0], gt_future[:, 1], 'go-', label='Ground Truth')

    # 绘制预测未来轨迹 (红色)
    plt.plot(pred_future[:, 0], pred_future[:, 1], 'rx--', label='Prediction')

    plt.legend()
    plt.title('Trajectory Prediction')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()

def collate_fn_with_meta(batch):
    histories = torch.stack([item[0] for item in batch]); futures = torch.stack([item[1] for item in batch]); metadata = [item[2] for item in batch]
    return histories, futures, metadata

# ----------------------------------
# 5. 主评估函数
# ----------------------------------
def main():
    print(f"Using device: {CONFIG['device']}")
    nusc = NuScenes(version=CONFIG['version'], dataroot=CONFIG['dataroot'], verbose=False)
    encoder = Encoder(CONFIG['input_dim'], CONFIG['hidden_dim'], CONFIG['n_layers'])
    decoder = Decoder(CONFIG['output_dim'], CONFIG['hidden_dim'], CONFIG['n_layers'])
    model = Seq2Seq(encoder, decoder, CONFIG['device']).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    print(f"Model loaded from {CONFIG['model_path']}")
    model.eval()
    # 我们需要在整个数据集上挖掘，而不仅仅是验证集
    # 如果数据集很大，可以考虑只用 val_dataset
    full_dataset = NuScenesTrajectoryDataset(nusc, CONFIG)

    # 使用整个数据集进行挖掘
    data_loader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4,
                             collate_fn=collate_fn_with_meta)

    # 加载模型


    ### NEW/MODIFIED ###
    # 用于存储每个场景中每个 agent 的失败帧
    # 结构: {scene_token: {instance_token: [(frame_index, fde_error), ...]}}
    failure_points = defaultdict(lambda: defaultdict(list))


    all_predictions = defaultdict(list)

    with torch.no_grad():
        for history, future, metadata in tqdm(data_loader, desc="Predicting and Calculating FDE"):
            history, future = history.to(CONFIG['device']), future.to(CONFIG['device'])
            output = model(history, future, 0)

            # 计算每个样本的 FDE
            fde_per_sample = torch.norm(output[:, -1, :] - future[:, -1, :], p=2, dim=1)

            for i in range(len(history)):
                meta = metadata[i]
                scene_token = meta["scene_token"]
                instance_token = meta["instance_token"]
                start_frame = meta["start_index_in_full_traj"]
                fde_error = fde_per_sample[i].item()

                ### MODIFICATION START ###
                # 模型输出已经是世界坐标，直接使用即可
                pred_future_coords = output[i].cpu().numpy()

                # 记录每个预测点的起始帧和对应的FDE
                all_predictions[(scene_token, instance_token)].append({
                    "start_frame": start_frame,
                    "fde_error": fde_error,
                    "pred_trajectory": pred_future_coords.tolist()  # 保存为 list
                })
    print(f"Finished prediction for all {len(all_predictions)} unique trajectories.")

    # 提前构建一个场景到所有agents的映射，避免在循环中重复查找，提高效率
    scene_to_agents_map = defaultdict(list)
    for scene_token, instance_token in full_dataset.full_trajectories.keys():
        scene_to_agents_map[scene_token].append(instance_token)
    print("Built scene-to-agents map.")

    # --- 第二步：识别关键事件并寻找交互对象 ---
    critical_event_index = defaultdict(list)
    n_context = CONFIG['critical_event_context_frames']
    fde_threshold = CONFIG['critical_event_threshold_fde']
    spike_threshold = CONFIG['critical_event_threshold_spike']

    # 从CONFIG获取交互参数
    proximity_thresh = CONFIG['interaction_proximity_threshold']
    heading_thresh_deg = CONFIG['interaction_heading_threshold_deg']

    for (scene_token, instance_token), agent_predictions in tqdm(all_predictions.items(),
                                                                 desc="Finding critical events"):
        if not agent_predictions:
            continue

        # 关键一步：必须按帧号排序，才能正确计算误差变化率
        agent_predictions.sort(key=lambda item: item["start_frame"])

        # 用于存储该 agent 已经识别出的事件中心帧，避免重复添加
        processed_event_frames = set()

        # 定义一个内部函数来处理事件的添加和交互分析，避免代码重复
        def process_and_add_event(event_data):
            center_frame = event_data["peak_fde_frame_in_traj"]
            if center_frame in processed_event_frames:
                return  # 避免重复处理

            full_traj_len = full_dataset.full_trajectories.get((scene_token, instance_token), np.array([])).shape[0]
            if full_traj_len == 0:
                return

            start_frame = max(0, center_frame - n_context)
            end_frame = min(full_traj_len, center_frame + n_context)

            if end_frame - start_frame <= (n_context / 2):
                return

            event_data["start_frame"] = start_frame
            event_data["end_frame"] = end_frame

            ### 核心交互分析部分 ###
            interactions_per_frame = {}
            all_agents_in_scene = scene_to_agents_map[scene_token]
            for frame_idx in range(start_frame, end_frame):
                interacting_tokens = find_interacting_agents(
                    key_agent_token=instance_token,
                    scene_token=scene_token,
                    frame_idx=frame_idx,
                    full_trajectories=full_dataset.full_trajectories,
                    all_agents_in_scene=all_agents_in_scene,
                    proximity_thresh=proximity_thresh,
                    heading_thresh_deg=heading_thresh_deg
                )
                if interacting_tokens:  # 只记录有交互对象的帧
                    interactions_per_frame[str(frame_idx)] = interacting_tokens
            event_data["interactions"] = interactions_per_frame
            critical_event_index[scene_token].append(event_data)
            processed_event_frames.add(center_frame)
        # --- 策略一：峰值误差法 (你原来的逻辑) ---
            # --- 策略一：峰值误差法 (修正后) ---
            if agent_predictions:
                peak_fde_event = max(agent_predictions, key=lambda item: item["fde_error"])
                peak_fde_frame = peak_fde_event["start_frame"]
                max_fde = peak_fde_event["fde_error"]
                pred_traj_for_peak = peak_fde_event["pred_trajectory"]

                if max_fde > fde_threshold:
                    event_center_frame = peak_fde_frame + CONFIG['history_len']
                    event_info = {
                        "reason": "peak_fde", "instance_token": instance_token,
                        "peak_fde_frame_in_traj": event_center_frame,
                        "value": round(max_fde, 2), "predicted_trajectory": pred_traj_for_peak
                    }
                    process_and_add_event(event_info)

        # --- 策略二：误差突增法 (新逻辑) ---
        if len(agent_predictions) > 1:
            max_spike, spike_event_frame, fde_at_spike = 0, -1, 0
            pred_traj_for_spike = None

            # 遍历寻找最大的FDE增量
            for i in range(1, len(agent_predictions)):
                current_pred = agent_predictions[i]
                prev_pred = agent_predictions[i - 1]

                if current_pred["start_frame"] == prev_pred["start_frame"] + 1:
                    spike = current_pred["fde_error"] - prev_pred["fde_error"]
                    if spike > max_spike:
                        max_spike = spike
                        spike_event_frame = current_pred["start_frame"]
                        fde_at_spike = current_pred["fde_error"]
                        pred_traj_for_spike = current_pred["pred_trajectory"]

            if max_spike > spike_threshold and spike_event_frame != -1:
                event_center_frame = spike_event_frame + CONFIG['history_len']
                event_info = {
                    "reason": "fde_spike", "instance_token": instance_token,
                    "peak_fde_frame_in_traj": event_center_frame,
                    "value": round(max_spike, 2), "fde_at_spike": round(fde_at_spike, 2),
                    "predicted_trajectory": pred_traj_for_spike
                }
                process_and_add_event(event_info)

        # --- 第三步：保存索引文件 (这部分不变) ---
        # ... (保存 JSON 文件的代码保持不变) ...
        # ... (打印统计信息的代码保持不变) ...

        # 打印更详细的统计信息
    output_path = CONFIG['critical_event_index_file']
    with open(output_path, 'w') as f:
        json.dump(critical_event_index, f, indent=4)

        # 打印更详细的统计信息
    peak_count = 0
    spike_count = 0
    for scene, events in critical_event_index.items():
        for event in events:
            if event['reason'] == 'peak_fde':
                peak_count += 1
            elif event['reason'] == 'fde_spike':
                spike_count += 1
    print(f"Index saved to: {output_path}")

    print(f"\n--- Critical Event Index Generation Finished ---")
    print(f"Index saved to: {CONFIG['critical_event_index_file']}")
    print(f"Total scenes with events: {len(critical_event_index)}")
    total_events = sum(len(v) for v in critical_event_index.values())
    print(f"Total event clips: {total_events}")
    print(f" - Found by Peak FDE: {peak_count}")
    print(f" - Found by FDE Spike: {spike_count}")


if __name__ == '__main__':
    main()