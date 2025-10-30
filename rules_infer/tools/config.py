# config.py
import torch


# config.py

class Config:
    # --- NuScenes 数据集参数 ---
    NUSCENES_DATA_ROOT = '/data0/senzeyu2/dataset/nuscenes'  # 【请修改】NuScenes数据集的根目录
    NUSCENES_VERSION = 'v1.0-trainval'  # 或者 'v1.0-mini'

    # --- 模型与轨迹参数 ---
    MODEL_SAVE_PATH = './trajectory_lstm.pth'  # 【请修改】您训练好的模型文件路径
    HIST_LEN = 8  # 历史轨迹长度 (对应 4s)
    PRED_LEN = 12  # 预测轨迹长度 (对应 6s)
    INPUT_DIM = 2  # 输入特征维度 (x, y)
    OUTPUT_DIM = 2  # 输出特征维度 (x, y)
    FPS = 2  # NuScenes 帧率

    # --- 事件检测阈值 (先用占位符，由 Part 1 分析后确定) ---
    FDE_THRESHOLD = 4.0  # 【待修改】例如，设为FDE的95百分位数
    ICE_PEAK_THRESHOLD = 5.0  # 【待修改】例如，设为Max ICE的95百分位数
    ICE_BASELINE = 1.0  # 【待修改】用于确定事件起止，可以设为中位数或稍高

    # --- 事件窗口与交互分析参数 ---
    PADDING_FRAMES_BEFORE = int(2.5 * FPS)  # 向前回溯 2.5 秒
    PADDING_FRAMES_AFTER = int(1.0 * FPS)  # 向后延伸 1.0 秒
    INTERACTION_RADIUS_M = 30.0  # 搜索交互Agent的半径（米）
    TOP_K_INTERACTING = 2  # 选取交互分数最高的K个agent

    # --- 误差分析脚本的输出路径 ---
    ANALYSIS_OUTPUT_PATH = './error_analysis_results'

    # --- 事件检测脚本的输出路径 ---
    EVENT_JSON_OUTPUT_PATH = './social_events.json'


