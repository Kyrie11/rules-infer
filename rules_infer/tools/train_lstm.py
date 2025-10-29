import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from nuscenes.nuscenes import NuScenes
import os
from tqdm import tqdm
from rules_infer.dataset.nuscenes import NuScenesTrajectoryDataset
from rules_infer.tools.motion_lstm import *

# --- 配置参数 ---
NUSCENES_PATH = '/data0/senzeyu2/dataset/nuscenes/'  # 修改为你的nuscenes数据集路径
NUSCENES_VERSION = 'v1.0-trainval'  # 你可以切换 'v1.0-trainval' 或 'v1.0-mini'
MODEL_SAVE_PATH = 'trajectory_lstm.pth'

# 训练超参数
HIST_LEN = 8
PRED_LEN = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10


class Config:
    # --- 数据集与路径 ---
    # !!! 修改为你的nuScenes数据集根目录 !!!
    NUSCENES_DATA_ROOT = '/data0/senzeyu2/dataset/nuscenes'
    NUSCENES_VERSION = 'v1.0-trainval'  # 使用mini数据集进行快速演示

    # --- 模型与训练参数 ---
    HIST_LEN = 8  # 历史轨迹长度 (N_in)
    PRED_LEN = 12  # 预测轨迹长度 (N_out)
    INPUT_DIM = 2  # 输入特征维度 (x, y)
    OUTPUT_DIM = 2  # 输出特征维度 (x, y)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20  # 演示目的，实际可增加
    MODEL_SAVE_PATH = 'trajectory_lstm.pth'

    # --- 事件检测与分析参数 ---
    FDE_THRESHOLD_M = 2.0  # 最终位移误差的绝对阈值（米）
    FDE_VEL_MULTIPLIER = 1.5  # FDE的相对阈值，FDE > 速度 * 这个乘数
    TTC_THRESHOLD_S = 4.0  # 触发交互分析的碰撞时间阈值（秒）

# --- 主函数 ---
def train_lstm(config, nusc):
    print("--- Starting LSTM Training ---")
    dataset = NuScenesTrajectoryDataset(config, nusc)
    if not dataset:
        print("Dataset is empty. Skipping training.")
        return

    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = TrajectoryLSTM(config)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0
        for history, future in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}"):
            optimizer.zero_grad()

            predictions = model(history)
            loss = criterion(predictions, future)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model trained and saved to {config.MODEL_SAVE_PATH}")

if __name__=="__main__":
    cfg = Config()
    print("Loading nuScenes dataset...")
    nusc = NuScenes(version=cfg.NUSCENES_VERSION, dataroot=cfg.NUSCENES_DATA_ROOT, verbose=False)
    train_lstm(cfg, nusc)