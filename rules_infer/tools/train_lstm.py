import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from rules_infer.dataset.nuscenes import NuScenesTrajectoryDataset
from rules_infer.tools.motion_lstm import *
from rules_infer.tools.config import Config


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