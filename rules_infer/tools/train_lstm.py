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

def train(model, iterator, optimizer, criterion, device, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    for i, (history, future, _) in enumerate(tqdm(iterator, desc="Training")):
        history = history.to(device)
        future = future.to(device)

        optimizer.zero_grad()

        # future 包含了我们希望模型预测的目标
        output = model(history, future, teacher_forcing_ratio)

        # output: [batch_size, future_len, output_dim]
        # future: [batch_size, future_len, output_dim]
        loss = criterion(output, future)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (history, future, _) in enumerate(tqdm(iterator, desc="Evaluating")):
            history = history.to(device)
            future = future.to(device)

            # 在评估时，不使用 teacher forcing，并且将 future 传入仅用于确定预测长度
            output = model(history, future, 0)  # teacher_forcing_ratio = 0

            loss = criterion(output, future)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# ----------------------------------
# 5. 主程序 (Main Program)
# ----------------------------------
def main():
    print(f"Using device: {CONFIG['device']}")

    # 1. 加载 NuScenes 数据
    if not os.path.exists(CONFIG['dataroot']):
        print(f"Error: NuScenes data root not found at {CONFIG['dataroot']}")
        print("Please download the NuScenes dataset and update the 'dataroot' in CONFIG.")
        return

    nusc = NuScenes(version=CONFIG['version'], dataroot=CONFIG['dataroot'], verbose=False)

    # 2. 创建数据集
    full_dataset = NuScenesTrajectoryDataset(nusc, CONFIG)

    if len(full_dataset) == 0:
        print("Dataset is empty. Check data path and processing logic.")
        return

    # 3. 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 4. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # 5. 初始化模型、优化器和损失函数
    encoder = Encoder(CONFIG['input_dim'], CONFIG['hidden_dim'], CONFIG['n_layers'])
    decoder = Decoder(CONFIG['output_dim'], CONFIG['hidden_dim'], CONFIG['n_layers'])
    model = Seq2Seq(encoder, decoder, CONFIG['device']).to(CONFIG['device'])

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()  # 均方误差损失，适用于回归问题

    # 6. 开始训练
    best_val_loss = float('inf')
    for epoch in range(CONFIG['n_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['n_epochs']} ---")

        train_loss = train(model, train_loader, optimizer, criterion, CONFIG['device'], CONFIG['teacher_forcing_ratio'])
        val_loss = evaluate(model, val_loader, criterion, CONFIG['device'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'nuscenes-lstm-model.pt')
            print("  -> Saved best model")

        print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')


if __name__ == '__main__':
    main()