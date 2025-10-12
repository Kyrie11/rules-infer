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

    # 在识别出的事件窗口前后额外扩展多少帧作为上下文
    'critical_event_context_frames': 10, # 往前和往后各扩展 10 帧 (即 5s)

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

    with torch.no_grad():
        for history, future, metadata in tqdm(data_loader, desc="Finding Failure Points"):
            history, future = history.to(CONFIG['device']), future.to(CONFIG['device'])
            output = model(history, future, 0)

            fde_per_sample = torch.norm(output[:, -1, :] - future[:, -1, :], p=2, dim=1)

            for i in range(len(history)):
                if fde_per_sample[i].item() > CONFIG['critical_event_threshold_fde']:
                    meta = metadata[i]
                    scene_token = meta["scene_token"]
                    instance_token = meta["instance_token"]
                    # 记录的是历史轨迹的起始帧
                    start_frame = meta["start_index_in_full_traj"]
                    fde_error = fde_per_sample[i].item()
                    failure_points[scene_token][instance_token].append((start_frame, fde_error))

    print(f"Found {sum(len(v) for v in failure_points.values())} scenes with potential critical events.")

    # --- 后处理：合并连续的失败帧为事件窗口 ---
    critical_event_index = defaultdict(list)
    n_context = CONFIG['critical_event_context_frames']
    hist_len = CONFIG['history_len']

    for scene_token, instances in tqdm(failure_points.items(), desc="Merging events"):
        for instance_token, points in instances.items():
            if not points:
                continue

            # 按帧索引排序
            points.sort()

            merged_events = []
            # 从第一个失败点开始
            current_start, current_max_fde = points[0]
            current_end = current_start

            for i in range(1, len(points)):
                frame_idx, fde = points[i]
                # 如果当前帧与上一个事件窗口接近（这里定义为在历史+未来长度内），则合并
                if frame_idx <= current_end + hist_len:
                    current_end = frame_idx
                    current_max_fde = max(current_max_fde, fde)
                else:
                    # 这是一个新事件，保存上一个事件
                    # 扩展上下文窗口
                    start_frame = max(0, current_start - n_context)
                    # 结束帧是最后一个触发点的历史+未来轨迹结束点，再加上下文
                    end_frame = current_end + hist_len + CONFIG['future_len']
                    # 获取轨迹总长度用于裁剪
                    full_traj_len = full_dataset.full_trajectories.get((scene_token, instance_token), []).shape[0]
                    end_frame = min(full_traj_len, end_frame + n_context)

                    merged_events.append({
                        "instance_token": instance_token,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "max_fde_in_event": round(current_max_fde, 2)
                    })
                    # 开始新的事件
                    current_start, current_max_fde = frame_idx, fde
                    current_end = current_start

            # 保存最后一个事件
            start_frame = max(0, current_start - n_context)
            end_frame = current_end + hist_len + CONFIG['future_len']
            full_traj_len = full_dataset.full_trajectories.get((scene_token, instance_token), []).shape[0]
            end_frame = min(full_traj_len, end_frame + n_context)

            merged_events.append({
                "instance_token": instance_token,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "max_fde_in_event": round(current_max_fde, 2)
            })

            critical_event_index[scene_token].extend(merged_events)

    # --- 保存索引文件 ---
    output_path = CONFIG['critical_event_index_file']
    with open(output_path, 'w') as f:
        json.dump(critical_event_index, f, indent=4)

    print(f"\n--- Critical Event Index Generation Finished ---")
    print(f"Index saved to: {output_path}")
    print(f"Total scenes with events: {len(critical_event_index)}")
    total_events = sum(len(v) for v in critical_event_index.values())
    print(f"Total event clips: {total_events}")


if __name__ == '__main__':
    main()