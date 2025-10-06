# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from config import Config
from ..dataset.nuscenes import NuscenesDataset
from motion_lstm import Seq2Seq


def main():
    # 1. 加载配置和初始化
    config = Config()
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. 加载NuScenes数据集
    nusc = NuScenes(version=config.VERSION, dataroot=config.DATAROOT, verbose=True)

    # 划分训练/验证集
    scene_splits = create_splits_scenes()
    train_scenes_names = scene_splits['mini_train'] if 'mini' in config.VERSION else scene_splits['train']
    # val_scenes_names = scene_splits['mini_val'] if 'mini' in config.VERSION else scene_splits['val']

    train_scenes = [s for s in nusc.scene if s['name'] in train_scenes_names]
    # val_scenes = [s for s in nusc.scene if s['name'] in val_scenes_names]

    # 3. 创建Dataset和DataLoader
    train_dataset = NuscenesDataset(nusc, train_scenes, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    print(f"Number of training sequences: {len(train_dataset)}")

    # 4. 初始化模型、损失函数和优化器
    model = Seq2Seq(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 5. 训练循环
    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            obs_traj = batch['obs_traj'].to(device)
            pred_traj_gt = batch['pred_traj_gt'].to(device)

            optimizer.zero_grad()

            # 模型需要src和target来进行teacher forcing
            pred_traj_fake = model(obs_traj, pred_traj_gt)

            loss = criterion(pred_traj_fake, pred_traj_gt)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

    # 保存模型
    torch.save(model.state_dict(), 'lstm_trajectory_predictor.pth')
    print("Model saved to lstm_trajectory_predictor.pth")


if __name__ == '__main__':
    main()

