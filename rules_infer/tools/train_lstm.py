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