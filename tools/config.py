# config.py
import torch


class Config:
    # --- Dataset Parameters ---
    DATAROOT = '/path/to/nuscenes/'  # <--- 修改为你的路径
    VERSION = 'v1.0-mini'

    # --- Trajectory Parameters ---
    OBS_LEN = 8
    PRED_LEN = 12
    SEQ_LEN = OBS_LEN + PRED_LEN

    # --- Feature Engineering ---
    # 我们的特征向量维度:
    # pos (x, y) = 2
    # vel (vx, vy) = 2
    # acc (ax, ay) = 2
    # is_on_drivable_area (1/0) = 1
    # traffic_light (one-hot: red, yellow, green, off) = 4
    # TOTAL INPUT_DIM = 2 + 2 + 2 + 1 + 4 = 11
    INPUT_DIM = 11

    # --- Training Parameters ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32  # 由于特征更复杂，可能需要减小batch size
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    # --- LSTM Model Parameters ---
    # 注意：INPUT_DIM 已在上面定义
    HIDDEN_DIM = 128  # 增加隐藏层维度以处理更复杂的输入
    NUM_LAYERS = 2
    OUTPUT_DIM = 2  # 输出仍然是 (x, y)
