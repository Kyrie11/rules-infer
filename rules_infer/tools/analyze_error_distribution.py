import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from rules_infer.tools.motion_lstm import TrajectoryLSTM
from rules_infer.tools.config import Config
# [在这里粘贴上面提供的 Encoder, Decoder, Seq2Seq 类的代码]
# ... (Encoder, Decoder, Seq2Seq class definitions go here) ...

# --- 配置参数 ---
NUSCENES_PATH = '/data0/senzeyu2/dataset/nuscenes'  # Nuscenes数据集的根目录
NUSCENES_VERSION = 'v1.0-trainval'  # 或者 'v1.0-mini'
MODEL_PATH = './trajectory_lstm.pth'  # 您训练好的模型文件路径
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型和轨迹参数
OBS_LEN = 8  # 观测长度 (对应 4s)
PRED_LEN = 12  # 预测长度 (对应 6s)
INPUT_DIM = 2  # 输入维度 (x, y)
OUTPUT_DIM = 2  # 输出维度 (x, y)
FPS = 2  # NuScenes的帧率


# --- 辅助函数 ---
def calculate_ice_signal(pred_traj, gt_traj, fps):
    """计算瞬时综合误差 (ICE) 信号"""
    # pred_traj, gt_traj: [PRED_LEN, 2]

    # 1. 瞬时位移误差 IDE(t)
    ide = np.linalg.norm(pred_traj - gt_traj, axis=1)

    # 2. 瞬时速度误差 IVE(t)
    # 速度是位置的差分, 乘以fps得到每秒的米数
    pred_vel = (pred_traj[1:] - pred_traj[:-1]) * fps
    gt_vel = (gt_traj[1:] - gt_traj[:-1]) * fps
    pred_speed = np.linalg.norm(pred_vel, axis=1)
    gt_speed = np.linalg.norm(gt_vel, axis=1)
    ive = np.abs(pred_speed - gt_speed)
    ive = np.append(ive, ive[-1])  # 补齐最后一位

    # 3. 瞬时加速度误差 IAE(t)
    pred_accel = (pred_vel[1:] - pred_vel[:-1]) * fps
    gt_accel = (gt_vel[1:] - gt_vel[:-1]) * fps
    pred_accel_scalar = np.linalg.norm(pred_accel, axis=1)
    gt_accel_scalar = np.linalg.norm(gt_accel, axis=1)
    iae = np.abs(pred_accel_scalar - gt_accel_scalar)
    iae = np.append(iae, [iae[-1], iae[-1]])  # 补齐最后两位

    # 4. 组合成ICE(t) - 权重可调
    w_pos = 1.0
    w_acc = 0.5  # 加速度权重很重要

    ice = w_pos * ide + w_acc * iae
    return ice


# --- 主逻辑 ---
def main():
    print(f"Using device: {DEVICE}")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_PATH, verbose=False)
    helper = PredictHelper(nusc)
    cfg = Config()
    # 加载模型
    model = TrajectoryLSTM(cfg).to(DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_fde, all_ade, all_max_ice = [], [], []

    # 获取'train_val' split中的所有验证场景
    val_scenes = [s for s in nusc.scene if nusc.get('log', s['log_token'])['logfile'].startswith('n008')]

    with torch.no_grad():
        for scene in tqdm(val_scenes, desc="Analyzing Scenes"):
            # 在场景中间点进行采样预测
            mid_sample_token = helper.get_sample_token_for_scene(scene['name'], 0.5)

            for instance_token in helper.get_annotations_for_scene(scene['name']):
                annotation = nusc.get('sample_annotation',
                                      nusc.get('instance', instance_token)['first_annotation_token'])
                if 'vehicle' not in annotation['category_name']:
                    continue

                past_traj = helper.get_past_for_agent(instance_token, mid_sample_token,
                                                      seconds=config.HIST_LEN / config.FPS, in_agent_frame=False)
                future_traj = helper.get_future_for_agent(instance_token, mid_sample_token,
                                                          seconds=config.PRED_LEN / config.FPS, in_agent_frame=False)

                if past_traj.shape[0] < config.HIST_LEN or future_traj.shape[0] < config.PRED_LEN:
                    continue

                obs_traj_tensor = torch.tensor(past_traj, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                # MODIFIED: 使用您的模型进行前向传播
                pred_future_traj_tensor = model(obs_traj_tensor)
                pred_future_traj = pred_future_traj_tensor.squeeze(0).cpu().numpy()

                fde = np.linalg.norm(pred_future_traj[-1] - future_traj[-1])
                ade = np.mean(np.linalg.norm(pred_future_traj - future_traj, axis=1))
                ice_signal = calculate_ice_signal(pred_future_traj, future_traj, config.FPS)
                max_ice = np.max(ice_signal)

                all_fde.append(fde)
                all_ade.append(ade)
                all_max_ice.append(max_ice)

    # 转换成numpy数组
    all_fde = np.array(all_fde)
    all_ade = np.array(all_ade)
    all_max_ice = np.array(all_max_ice)
    os.makedirs(cfg.ANALYSIS_OUTPUT_PATH, exist_ok=True)
    # 绘制直方图并打印百分位数
    print("\n--- Error Metrics Distribution Analysis ---")
    for name, data in [("FDE", all_fde), ("ADE", all_ade), ("Max ICE", all_max_ice)]:
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=100, range=(0, np.percentile(data, 99.5)))  # 忽略极端离群值以获得更好的可视化
        plt.title(f"Distribution of {name}")
        plt.xlabel("Error Value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(cfg.ANALYSIS_OUTPUT_PATH, f'{name}_distribution.png'))
        plt.close()

        print(f"\nStatistics for {name}:")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  50th Percentile (Median): {np.percentile(data, 50):.2f}")
        print(f"  90th Percentile: {np.percentile(data, 90):.2f}")
        print(f"  95th Percentile: {np.percentile(data, 95):.2f}")
        print(f"  99th Percentile: {np.percentile(data, 99):.2f}")


if __name__ == '__main__':
    main()

