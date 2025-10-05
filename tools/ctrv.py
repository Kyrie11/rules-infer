import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


def get_instance_history(nusc: NuScenes, scene_token: str, instance_token: str):
    """获取一个instance在整个scene中的所有annotation和sample信息"""
    history = []
    scene = nusc.get('scene', scene_token)
    sample_token = scene['first_sample_token']

    ann_token = nusc.get('sample', sample_token)['anns'][0]
    # 找到该instance的第一个annotation
    first_ann = nusc.get('sample_annotation', ann_token)
    while first_ann['instance_token'] != instance_token and first_ann['next']:
        first_ann = nusc.get('sample_annotation', first_ann['next'])

    if first_ann['instance_token'] != instance_token:
        return []  # 该instance不在第一个sample中出现

    curr_ann = first_ann
    while curr_ann:
        ann_data = nusc.get('sample_annotation', curr_ann['token'])
        sample_data = nusc.get('sample', ann_data['sample_token'])
        history.append({'ann': ann_data, 'sample': sample_data})
        if not ann_data['next']:
            break
        curr_ann = nusc.get('sample_annotation', ann_data['next'])
    return history


def ctrv_motion_model(x, dt):
    """CTRV 状态转移函数 f(x)"""
    x_new = x.copy()
    px, py, v, yaw, yaw_rate = x

    if abs(yaw_rate) > 1e-4:  # 避免除以零
        radius = v / yaw_rate
        x_new[0] = px + radius * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
        x_new[1] = py + radius * (-np.cos(yaw + yaw_rate * dt) + np.cos(yaw))
        x_new[3] = (yaw + yaw_rate * dt) % (2 * np.pi)  # 角度归一化
    else:  # 近似为直线运动
        x_new[0] = px + v * np.cos(yaw) * dt
        x_new[1] = py + v * np.sin(yaw) * dt
    # v 和 yaw_rate 保持不变
    return x_new


def measurement_function(x):
    """观测函数 h(x)，状态直接映射到观测"""
    return x[:2]  # 只观测位置 (px, py)


def calculate_ctrv_surprise(instance_history):
    """使用CTRV卡尔曼滤波器计算一个instance轨迹的惊讶度分数"""
    if len(instance_history) < 3:
        return {}

    # 状态向量 [x, y, v, yaw, yaw_rate]
    dt = (instance_history[1]['sample']['timestamp'] - instance_history[0]['sample']['timestamp']) * 1e-6

    # UKF 初始化
    points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=0.1)
    kf = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=dt, hx=measurement_function, fx=ctrv_motion_model, points=points)

    # 初始状态估计
    ann0 = instance_history[0]['ann']
    ann1 = instance_history[1]['ann']
    pos0 = ann0['translation'][:2]
    pos1 = ann1['translation'][:2]

    v = np.linalg.norm(pos1 - pos0) / dt
    yaw0 = Quaternion(ann0['rotation']).yaw_pitch_roll[0]
    yaw1 = Quaternion(ann1['rotation']).yaw_pitch_roll[0]
    yaw_rate = (yaw1 - yaw0) / dt

    kf.x = np.array([pos0[0], pos0[1], v, yaw0, yaw_rate])
    kf.P *= 0.1  # 初始不确定性
    kf.R = np.diag([0.5, 0.5])  # 观测噪声
    kf.Q = np.diag([0.1, 0.1, 0.2, 0.01, 0.01])  # 过程噪声

    surprise_scores = {}

    for i in range(1, len(instance_history) - 1):
        # 预测
        kf.predict()

        # 更新
        current_ann = instance_history[i]['ann']
        z = current_ann['translation'][:2]  # 当前真实观测
        kf.update(z)

        # 用更新后的状态，预测下一帧
        kf.predict()
        predicted_pos = kf.x_prior[:2]  # 预测的下一帧位置

        # 获取下一帧的真实位置
        next_ann = instance_history[i + 1]['ann']
        actual_pos = next_ann['translation'][:2]

        # 计算预测误差（惊讶度）
        prediction_error = np.linalg.norm(predicted_pos - actual_pos)

        sample_token = current_ann['sample_token']
        surprise_scores[sample_token] = float(prediction_error)

        # 更新dt和模型状态，准备下一次循环
        if i < len(instance_history) - 2:
            dt = (instance_history[i + 2]['sample']['timestamp'] - instance_history[i + 1]['sample'][
                'timestamp']) * 1e-6
            kf.dt = dt

            # 从真实数据中更新速度和角速度，以便下一次预测更准确
            v = np.linalg.norm(actual_pos - z) / dt
            yaw_curr = Quaternion(current_ann['rotation']).yaw_pitch_roll[0]
            yaw_next = Quaternion(next_ann['rotation']).yaw_pitch_roll[0]
            yaw_rate = (yaw_next - yaw_curr) / dt

            kf.x[2] = v
            kf.x[3] = yaw_next  # 使用最新的yaw
            kf.x[4] = yaw_rate

    return surprise_scores