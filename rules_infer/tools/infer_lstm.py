import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import random
# 确保 nuscenes-devkit 和其他库已安装
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from rules_infer.dataset.nuscenes import NuScenesTrajectoryDataset
from rules_infer.tools.motion_lstm import Encoder, Decoder, Seq2Seq

CONFIG = {
    # --- 数据和模型路径 ---
    'dataroot': '/data0/senzeyu2/dataset/',  # <--- !!! 修改这里 !!!
    'version': 'v1.0-trainval',  # 建议先用 'v1.0-mini' 测试，然后换成 'v1.0-trainval'
    'model_path': '../../nuscenes-lstm-model.pt',  # 你保存的模型权重文件
    'output_dir': 'eval_results',  # 保存可视化结果的文件夹

    # --- 模型和数据参数 (必须与训练时一致) ---
    'history_len': 8,
    'future_len': 12,
    'input_dim': 4,  # (x, y, is_near_tl, dist_to_tl) - 如果训练时没用地图，这里是 2
    'hidden_dim': 128,
    'output_dim': 2,
    'n_layers': 2,

    # --- 评估参数 ---
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # --- 地图参数 (如果训练时使用了) ---
    'traffic_light_distance_threshold': 30.0,
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


# ----------------------------------
# 5. 主评估函数
# ----------------------------------
def main():
    print(f"Using device: {CONFIG['device']}")

    # 创建输出文件夹
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 1. 加载 NuScenes 数据
    nusc = NuScenes(version=CONFIG['version'], dataroot=CONFIG['dataroot'], verbose=False)

    # 2. 创建数据集
    full_dataset = NuScenesTrajectoryDataset(nusc, CONFIG)

    # 3. 划分训练集和验证集 (使用固定随机种子确保与训练时划分一致)
    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Using validation set of size: {len(val_dataset)}")

    # 4. 创建 DataLoader
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # 5. 初始化模型并加载权重
    encoder = Encoder(CONFIG['input_dim'], CONFIG['hidden_dim'], CONFIG['n_layers'])
    decoder = Decoder(CONFIG['output_dim'], CONFIG['hidden_dim'], CONFIG['n_layers'])
    model = Seq2Seq(encoder, decoder, CONFIG['device']).to(CONFIG['device'])

    try:
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print(f"Model weights loaded successfully from {CONFIG['model_path']}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {CONFIG['model_path']}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # 6. 开始评估
    model.eval()
    total_ade = []
    total_fde = []

    with torch.no_grad():
        for batch_idx, (history, future) in enumerate(tqdm(val_loader, desc="Evaluating")):
            history = history.to(CONFIG['device'])
            future = future.to(CONFIG['device'])

            # 预测时 teacher_forcing_ratio 必须为 0
            output = model(history, future, 0)

            # 计算指标
            batch_ade = calculate_ade(output, future)
            batch_fde = calculate_fde(output, future)
            total_ade.append(batch_ade.item())
            total_fde.append(batch_fde.item())

            # 可视化该 batch 的第一个样本 (每隔几个 batch 可视化一次以防图片过多)
            if batch_idx % 20 == 0:
                save_path = os.path.join(CONFIG['output_dir'], f'prediction_batch{batch_idx}_sample0.png')
                # history 的输入维度是4, 但绘图只需要前两维 (x,y)
                visualize_prediction(history[0, :, :2], future[0], output[0], save_path)

    # 7. 打印最终结果
    avg_ade = np.mean(total_ade)
    avg_fde = np.mean(total_fde)

    print("\n--- Evaluation Finished ---")
    print(f"Average Displacement Error (ADE): {avg_ade:.4f} meters")
    print(f"Final Displacement Error (FDE):   {avg_fde:.4f} meters")
    print(f"Visualization results are saved in '{CONFIG['output_dir']}' directory.")


if __name__ == '__main__':
    main()