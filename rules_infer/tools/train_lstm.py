import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from nuscenes.nuscenes import NuScenes
import os
from tqdm import tqdm
from rules_infer.dataset.nuscenes import NuScenesTrajectoryDataset
from rules_infer.tools.motion_lstm import *
# ----------------------------------
# 1. 配置参数 (Configuration)
# ----------------------------------
CONFIG = {
    # 数据集路径，请修改为你自己的路径
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <--- !!! 修改这里 !!!
    'version': 'v1.0-trainval',  # 先用 'v1.0-mini' 测试, 跑通后再换成 'v1.0-trainval'

    # 轨迹参数 (以 2Hz 的采样率计算)
    'history_len': 8,  # 使用 4s 的历史轨迹 (8 * 0.5s)
    'future_len': 12,  # 预测 6s 的未来轨迹 (12 * 0.5s)

    # 模型参数
    'input_dim': 4,  # 输入维度: (x, y)
    'hidden_dim': 64,  # LSTM 隐藏层维度
    'output_dim': 2,  # 输出维度: (x, y)
    'n_layers': 2,  # LSTM 层数

    # 训练参数
    'batch_size': 64,
    'n_epochs': 50,
    'learning_rate': 0.001,
    'teacher_forcing_ratio': 0.5,  # 训练时使用真实值作为下一步输入的概率
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'traffic_light_distance_threshold': 30.0
}


def get_heading(points):
    """从轨迹的最后两点计算朝向角"""
    p1, p2 = points[-2], points[-1]
    if torch.all(p1 == p2): return 0.0
    return torch.atan2(p2[1] - p1[1], p2[0] - p1[0])


def transform_to_agent_centric(history, future):
    """将一个样本的历史和未来轨迹转换到agent中心坐标系"""
    anchor_point = history[-1, :2].clone()
    anchor_yaw = get_heading(history[:, :2])

    rot_matrix = torch.tensor([
        [torch.cos(anchor_yaw), -torch.sin(anchor_yaw)],
        [torch.sin(anchor_yaw), torch.cos(anchor_yaw)]
    ])

    # 转换历史轨迹坐标
    history[:, :2] = (history[:, :2] - anchor_point) @ rot_matrix.T
    # 转换未来轨迹坐标
    future[:, :2] = (future[:, :2] - anchor_point) @ rot_matrix.T

    return history, future


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    for history_global, future_global in tqdm(dataloader, desc="Training"):
        history_global, future_global = history_global.to(device), future_global.to(device)

        # 对batch中的每个样本进行坐标系转换
        batch_size = history_global.shape[0]
        history_centric = torch.zeros_like(history_global)
        future_centric = torch.zeros_like(future_global)

        for i in range(batch_size):
            history_centric[i], future_centric[i] = transform_to_agent_centric(
                history_global[i], future_global[i]
            )

        optimizer.zero_grad()

        # 注意这里的trg是future_centric
        output = model(history_centric, future_centric, teacher_forcing_ratio)

        loss = criterion(output, future_centric)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # 梯度裁剪
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for history_global, future_global in tqdm(dataloader, desc="Evaluating"):
            history_global, future_global = history_global.to(device), future_global.to(device)

            batch_size = history_global.shape[0]
            history_centric = torch.zeros_like(history_global)
            future_centric = torch.zeros_like(future_global)

            for i in range(batch_size):
                history_centric[i], future_centric[i] = transform_to_agent_centric(
                    history_global[i], future_global[i]
                )

            # 评估时关闭 teacher forcing
            output = model(history_centric, future_centric, 0)

            loss = criterion(output, future_centric)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


if __name__ == '__main__':
    # --- 配置 ---
    config = {
        'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <<<--- 修改为你的nuScenes路径
        'version': 'v1.0-trainval',
        'history_len': 8,  # 4s
        'future_len': 12,  # 6s
        'traffic_light_distance_threshold': 30.0,
        # 训练参数
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 20,
        'val_split': 0.15,
        'teacher_forcing_ratio': 0.5,
        # 模型参数
        'input_dim': 4,  # [x, y, is_near_tl, dist_to_tl]
        'output_dim': 2,  # [x, y]
        'hidden_dim': 64,
        'n_layers': 2,
        'model_save_path': 'seq2seq_model.pth'
    }

    # --- 初始化 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    nusc = NuScenes(version=config['version'], dataroot=config['dataroot'], verbose=False)

    # --- 数据加载 ---
    full_dataset = NuScenesTrajectoryDataset(nusc, config)

    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # --- 模型、优化器、损失函数 ---
    encoder = Encoder(config['input_dim'], config['hidden_dim'], config['n_layers'])
    decoder = Decoder(config['output_dim'], config['hidden_dim'], config['n_layers'])
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()  # 均方误差损失，适合坐标预测

    # --- 训练循环 ---
    best_valid_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config['teacher_forcing_ratio'])
        valid_loss = evaluate_epoch(model, val_loader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Validation loss improved. Model saved to {config['model_save_path']}")

        print(f'Epoch {epoch + 1}:')
        print(f'\tTrain Loss: {train_loss:.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f}')

    print("\nTraining complete!")
    print(f"Best model saved at '{config['model_save_path']}' with validation loss: {best_valid_loss:.4f}")