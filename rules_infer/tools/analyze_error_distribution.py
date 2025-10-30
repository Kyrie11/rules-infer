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


def get_sample_token_by_index(nusc, scene, sample_idx):
    """
    根据索引（第几帧）获取场景中的sample_token。
    """
    if sample_idx < 0 or sample_idx >= scene['nbr_samples']:
        raise IndexError("Sample index out of bounds.")

    current_token = scene['first_sample_token']
    for _ in range(sample_idx):
        sample = nusc.get('sample', current_token)
        current_token = sample['next']
        if not current_token:  # 以防万一
            break
    return current_token

def get_all_instances_in_scene(nusc, scene):
    """
    正确的方法：遍历场景中的所有样本来收集所有唯一的instance_token。
    """
    instance_tokens = set()
    current_sample_token = scene['first_sample_token']
    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            instance_tokens.add(ann['instance_token'])
        current_sample_token = sample['next']
    return list(instance_tokens)

# --- 主逻辑 ---
def main():
    config = Config()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    nusc = NuScenes(version=config.NUSCENES_VERSION, dataroot=config.NUSCENES_DATA_ROOT, verbose=False)
    helper = PredictHelper(nusc)

    model = TrajectoryLSTM(config).to(DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_fde, all_ade, all_max_ice = [], [], []

    val_scenes = [s for s in nusc.scene if nusc.get('log', s['log_token'])['logfile'].startswith('n008')]

    with torch.no_grad():
        for scene in tqdm(val_scenes, desc="Analyzing Scenes"):
            mid_index = scene['nbr_samples'] // 2
            try:
                mid_sample_token = get_sample_token_by_index(nusc, scene, mid_index)
            except IndexError:
                continue

            # --- CORRECTED LOGIC to get all instances ---
            all_instance_tokens_in_scene = get_all_instances_in_scene(nusc, scene)

            for instance_token in all_instance_tokens_in_scene:
                try:
                    annotation = nusc.get('sample_annotation',
                                          nusc.get('instance', instance_token)['first_annotation_token'])
                    if 'vehicle' not in annotation['category_name']:
                        continue
                except KeyError:
                    continue

                past_traj = helper.get_past_for_agent(instance_token, mid_sample_token,
                                                      seconds=config.HIST_LEN / config.FPS, in_agent_frame=False)
                future_traj = helper.get_future_for_agent(instance_token, mid_sample_token,
                                                          seconds=config.PRED_LEN / config.FPS, in_agent_frame=False)

                if past_traj.shape[0] < config.HIST_LEN or future_traj.shape[0] < config.PRED_LEN:
                    continue

                obs_traj_tensor = torch.tensor(past_traj, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                pred_future_traj_tensor = model(obs_traj_tensor)
                pred_future_traj = pred_future_traj_tensor.squeeze(0).cpu().numpy()

                fde = np.linalg.norm(pred_future_traj[-1] - future_traj[-1])
                ade = np.mean(np.linalg.norm(pred_future_traj - future_traj, axis=1))
                ice_signal = calculate_ice_signal(pred_future_traj, future_traj, config.FPS)
                max_ice = np.max(ice_signal)

                all_fde.append(fde)
                all_ade.append(ade)
                all_max_ice.append(max_ice)

    # ... 后续的绘图和统计代码保持不变 ...
    all_fde, all_ade, all_max_ice = np.array(all_fde), np.array(all_ade), np.array(all_max_ice)
    os.makedirs(config.ANALYSIS_OUTPUT_PATH, exist_ok=True)
    print("\n--- Error Metrics Distribution Analysis ---")
    for name, data in [("FDE", all_fde), ("ADE", all_ade), ("Max_ICE", all_max_ice)]:
        data = data[np.isfinite(data)]
        if len(data) == 0: continue
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=100, range=(0, np.percentile(data, 99.5)))
        plt.title(f"Distribution of {name}")
        plt.xlabel("Error Value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(config.ANALYSIS_OUTPUT_PATH, f'{name}_distribution.png'))
        plt.close()
        print(f"\nStatistics for {name}:")
        print(
            f"  Mean: {np.mean(data):.2f}, 50th: {np.percentile(data, 50):.2f}, 90th: {np.percentile(data, 90):.2f}, 95th: {np.percentile(data, 95):.2f}, 99th: {np.percentile(data, 99):.2f}")
    print(f"\nAnalysis complete. Histograms saved to '{config.ANALYSIS_OUTPUT_PATH}' directory.")
    print("Please update the threshold values in 'config.py' based on these results.")


if __name__ == '__main__':
    main()

