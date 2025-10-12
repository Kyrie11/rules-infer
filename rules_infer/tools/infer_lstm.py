import torch
from torch.utils.data import DataLoader, random_split
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
# 确保 nuscenes-devkit 和其他库已安装
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from collections import defaultdict

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

    # 保存最终索引文件的路径
    'critical_event_index_file': 'critical_events.json'
}


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

    # 我们需要在整个数据集上挖掘，而不仅仅是验证集
    # 如果数据集很大，可以考虑只用 val_dataset
    full_dataset = NuScenesTrajectoryDataset(nusc, CONFIG)

    # 使用整个数据集进行挖掘
    data_loader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4,
                             collate_fn=collate_fn_with_meta)

    # 加载模型
    encoder = Encoder(CONFIG['input_dim'], CONFIG['hidden_dim'], CONFIG['n_layers'])
    decoder = Decoder(CONFIG['output_dim'], CONFIG['hidden_dim'], CONFIG['n_layers'])
    model = Seq2Seq(encoder, decoder, CONFIG['device']).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    print(f"Model loaded from {CONFIG['model_path']}")
    model.eval()

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

                # 记录每个预测点的起始帧和对应的FDE
                all_predictions[(scene_token, instance_token)].append((start_frame, fde_error))

    print(f"Finished prediction for all {len(all_predictions)} unique trajectories.")

    # --- 第二步：按你的新逻辑处理，找到每个 agent 的最差预测点 ---
    critical_event_index = defaultdict(list)
    n_context = CONFIG['critical_event_context_frames']
    fde_threshold = CONFIG['critical_event_threshold_fde']
    spike_threshold = CONFIG['critical_event_threshold_spike']

    for (scene_token, instance_token), agent_predictions in tqdm(all_predictions.items(),
                                                                 desc="Finding critical events"):
        if not agent_predictions:
            continue

        # 关键一步：必须按帧号排序，才能正确计算误差变化率
        agent_predictions.sort(key=lambda item: item[0])

        # 用于存储该 agent 已经识别出的事件中心帧，避免重复添加
        processed_event_frames = set()

        # --- 策略一：峰值误差法 (你原来的逻辑) ---
        # 找到 FDE 最大的那个预测点
        peak_fde_event = max(agent_predictions, key=lambda item: item[1])
        peak_fde_frame, max_fde = peak_fde_event

        if max_fde > fde_threshold:
            event_center_frame = peak_fde_frame + CONFIG['history_len']

            # 检查是否已处理过
            if event_center_frame not in processed_event_frames:
                # 添加到索引
                full_traj_len = full_dataset.full_trajectories.get((scene_token, instance_token), np.array([])).shape[0]
                if full_traj_len > 0:
                    start_frame = max(0, event_center_frame - n_context)
                    end_frame = min(full_traj_len, event_center_frame + n_context)

                    if end_frame - start_frame > (n_context / 2):
                        critical_event_index[scene_token].append({
                            "reason": "peak_fde",  # 标明原因
                            "instance_token": instance_token,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "peak_fde_frame_in_traj": event_center_frame,
                            "value": round(max_fde, 2)  # 记录峰值FDE
                        })
                        processed_event_frames.add(event_center_frame)

        # --- 策略二：误差突增法 (新逻辑) ---
        if len(agent_predictions) > 1:
            max_spike = 0
            spike_event_frame = -1
            fde_at_spike = 0

            # 遍历寻找最大的FDE增量
            for i in range(1, len(agent_predictions)):
                current_frame, current_fde = agent_predictions[i]
                prev_frame, prev_fde = agent_predictions[i - 1]

                # 确保是连续的帧预测，如果中间有跳跃则不计算spike
                if current_frame == prev_frame + 1:
                    spike = current_fde - prev_fde
                    if spike > max_spike:
                        max_spike = spike
                        # "事件"发生在误差突增之后的那一帧
                        spike_event_frame = current_frame
                        fde_at_spike = current_fde

            if max_spike > spike_threshold:
                event_center_frame = spike_event_frame + CONFIG['history_len']

                # 检查是否已处理过
                if event_center_frame not in processed_event_frames:
                    # 添加到索引
                    full_traj_len = \
                    full_dataset.full_trajectories.get((scene_token, instance_token), np.array([])).shape[0]
                    if full_traj_len > 0:
                        start_frame = max(0, event_center_frame - n_context)
                        end_frame = min(full_traj_len, event_center_frame + n_context)

                        if end_frame - start_frame > (n_context / 2):
                            critical_event_index[scene_token].append({
                                "reason": "fde_spike",  # 标明原因
                                "instance_token": instance_token,
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "peak_fde_frame_in_traj": event_center_frame,
                                "value": round(max_spike, 2),  # 记录spike值
                                "fde_at_spike": round(fde_at_spike, 2)  # 记录spike发生时的FDE
                            })
                            processed_event_frames.add(event_center_frame)

        # --- 第三步：保存索引文件 (这部分不变) ---
        # ... (保存 JSON 文件的代码保持不变) ...
        # ... (打印统计信息的代码保持不变) ...

        # 打印更详细的统计信息
    peak_count = 0
    spike_count = 0
    for scene, events in critical_event_index.items():
        for event in events:
            if event['reason'] == 'peak_fde':
                peak_count += 1
            elif event['reason'] == 'fde_spike':
                spike_count += 1

    print(f"\n--- Critical Event Index Generation Finished ---")
    print(f"Index saved to: {CONFIG['critical_event_index_file']}")
    print(f"Total scenes with events: {len(critical_event_index)}")
    total_events = sum(len(v) for v in critical_event_index.values())
    print(f"Total event clips: {total_events}")
    print(f" - Found by Peak FDE: {peak_count}")
    print(f" - Found by FDE Spike: {spike_count}")


if __name__ == '__main__':
    main()