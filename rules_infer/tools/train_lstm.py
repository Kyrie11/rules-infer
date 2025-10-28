import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from nuscenes.nuscenes import NuScenes
import os
from tqdm import tqdm
from rules_infer.dataset.nuscenes import NuScenesTrajectoryDataset
from rules_infer.tools.motion_lstm import *

NUSCENES_PATH = '/data0/senzeyu2/dataset/nuscenes'  # 修改为你的nuscenes数据集路径
NUSCENES_VERSION = 'v1.0-trainval'  # 使用mini版本进行快速测试，或 'v1.0-trainval' 进行完整训练
MODEL_SAVE_PATH = 'trajectory_lstm.pth'

# 训练超参数
HIST_LEN = 8
PRED_LEN = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10


# --- 主函数 ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载NuScenes数据集
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_PATH, verbose=True)

    # 2. 创建Dataset和DataLoader
    train_dataset = NuScenesTrajectoryDataset(nusc, hist_len=HIST_LEN, pred_len=PRED_LEN, split=NUSCENES_VERSION)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 3. 初始化模型、损失函数和优化器
    model = TrajectoryLSTM(pred_len=PRED_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练循环
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (history, future_gt) in enumerate(train_loader):
            history = history.to(device)
            future_gt = future_gt.to(device)

            # 前向传播
            future_pred = model(history)

            # 计算损失
            loss = criterion(future_pred, future_gt)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch + 1}/{EPOCHS}] finished. Average Loss: {total_loss / len(train_loader):.4f}')

    # 5. 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()