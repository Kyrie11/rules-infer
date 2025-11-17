import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment # 用于匈牙利匹配
from scipy.signal import savgol_filter


# ----------------------------------------------------
# 追踪器 (SimpleTrack/Kalman Logic)
# ----------------------------------------------------
class KalmanBoxTracker(object):
    """
    追踪单个目标的卡尔曼滤波器
    状态向量 [x, y, yaw, l, w, vx, vy, yaw_rate] (8维)
    观测向量 [x, y, yaw, l, w] (5维)
    """
    count = 0

    def __init__(self, detection_info):
        # detection_info = (x, y, l, w, yaw)
        self.kf = KalmanFilter(dim_x=8, dim_z=5)

        # 状态转移矩阵 F
        self.kf.F = np.eye(8)
        # (在 predict 步骤中我们会动态更新 dt)

        # 观测矩阵 H
        self.kf.H = np.zeros((5, 8))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 2] = 1  # yaw
        self.kf.H[3, 3] = 1  # l
        self.kf.H[4, 4] = 1  # w

        # 初始协方差 P, 过程噪声 Q, 测量噪声 R (需要精调)
        self.kf.P *= 10.
        self.kf.Q[5:, 5:] *= 0.1  # 速度/转向率的噪声
        self.kf.R *= 1.

        # 初始化状态
        self.kf.x[0] = detection_info[0]  # x
        self.kf.x[1] = detection_info[1]  # y
        self.kf.x[2] = detection_info[4]  # yaw
        self.kf.x[3] = detection_info[2]  # l
        self.kf.x[4] = detection_info[3]  # w

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # 存储历史用于分析
        self.history = []
        self.last_timestamp = -1

    def update_dt(self, dt):
        """根据真实时间差更新状态转移矩阵 F"""
        self.kf.F[0, 5] = dt
        self.kf.F[1, 6] = dt
        self.kf.F[2, 7] = dt

    def predict(self, timestamp):
        """预测"""
        if self.last_timestamp > 0:
            dt = timestamp - self.last_timestamp
            self.update_dt(dt)
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, detection_info, timestamp):
        """更新"""
        # detection_info = (x, y, l, w, yaw)
        z = np.array([detection_info[0],
                      detection_info[1],
                      detection_info[4],
                      detection_info[2],
                      detection_info[3]])

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_timestamp = timestamp  # 更新时间戳
        self.kf.update(z)

    def get_state(self):
        """获取当前状态 [x, y, l, w, yaw, vx, vy]"""
        return np.array([
            self.kf.x[0], self.kf.x[1], self.kf.x[3],
            self.kf.x[4], self.kf.x[2], self.kf.x[5], self.kf.x[6]
        ]).reshape(7)


class GlobalTracker(object):
    def __init__(self, dist_threshold=2.0, max_age=5):
        self.trackers = []
        self.dist_threshold = dist_threshold
        self.max_age = max_age  # 允许 tracker 丢失的最大帧数

    def update(self, detections, timestamp):
        # 1. 预测所有 tracker 到当前帧
        predicted_states = []
        for trk in self.trackers:
            state = trk.predict(timestamp)
            predicted_states.append(state)

        # 2. 关联: detections 和 predicted_states
        if len(predicted_states) > 0 and len(detections) > 0:
            dist_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)
            for d, det in enumerate(detections):
                for t, state in enumerate(predicted_states):
                    dist_matrix[d, t] = np.linalg.norm(det[:2] - state[:2])  # 仅匹配 x, y

            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            matched_indices = []
            for r, c in zip(row_ind, col_ind):
                if dist_matrix[r, c] < self.dist_threshold:
                    matched_indices.append((r, c))

            matched_det_indices = [m[0] for m in matched_indices]
            matched_trk_indices = [m[1] for m in matched_indices]

            unmatched_det_indices = [d for d in range(len(detections)) if d not in matched_det_indices]
            unmatched_trk_indices = [t for t in range(len(self.trackers)) if t not in matched_trk_indices]

        else:
            unmatched_det_indices = list(range(len(detections)))
            unmatched_trk_indices = list(range(len(self.trackers)))

        # 3. 更新
        # 3.1 更新匹配上的 trackers
        for det_idx, trk_idx in matched_indices:
            self.trackers[trk_idx].update(detections[det_idx], timestamp)

        # 3.2 创建新的 trackers
        for idx in unmatched_det_indices:
            new_tracker = KalmanBoxTracker(detections[idx])
            new_tracker.last_timestamp = timestamp
            self.trackers.append(new_tracker)

        # 3.3 删除旧的 trackers
        active_trackers = []
        for i, trk in enumerate(self.trackers):
            if trk.time_since_update <= self.max_age:
                active_trackers.append(trk)
        self.trackers = active_trackers

        # 只返回那些被“激活”的 (例如，连续命中N帧的)
        return [trk for trk in self.trackers if trk.hit_streak > 2]


class KinematicsAnalyzer:
    """计算所有你需要的“其他信息”"""

    def __init__(self, window_length=9, polyorder=2):
        if window_length % 2 == 0 or window_length < 3:
            raise ValueError("window_length 必须是大于1的奇数")
        self.window_length = window_length
        self.polyorder = polyorder

    def calculate_kinematics(self, track_history):
        """
        从 tracker 的历史中计算平滑的运动学。
        参数:
        track_history: list, 包含 (timestamp, x, y, yaw, vx_kalman, vy_kalman)
        """

        if len(track_history) < self.window_length:
            latest_state = track_history[-1]
            return {
                'timestamp': latest_state[0],
                'x': latest_state[1], 'y': latest_state[2], 'yaw': latest_state[3],
                'vx': latest_state[4], 'vy': latest_state[5],
                'speed': np.sqrt(latest_state[4] ** 2 + latest_state[5] ** 2),
                'ax': 0, 'ay': 0, 'acceleration': 0, 'jerk': 0  # 无法计算
            }

        # 1. 提取时间序列
        timestamps = np.array([h[0] for h in track_history])
        vx_kalman = np.array([h[4] for h in track_history])  # 使用卡尔曼平滑后的速度
        vy_kalman = np.array([h[5] for h in track_history])

        dt_array = np.gradient(timestamps)

        # 3. 平滑速度
        vx_smooth = savgol_filter(vx_kalman, self.window_length, self.polyorder, deriv=0)
        vy_smooth = savgol_filter(vy_kalman, self.window_length, self.polyorder, deriv=0)

        # 4. 计算加速度
        ax_smooth = np.gradient(vx_smooth, dt_array)
        ay_smooth = np.gradient(vy_smooth, dt_array)
        # 再次平滑加速度 (可选，但推荐)
        ax_smooth = savgol_filter(ax_smooth, self.window_length, self.polyorder, deriv=0)
        ay_smooth = savgol_filter(ay_smooth, self.window_length, self.polyorder, deriv=0)

        # 5. 计算 Jerk
        jx_smooth = np.gradient(ax_smooth, dt_array)
        jy_smooth = np.gradient(ay_smooth, dt_array)

        # 6. 组装最新一帧的结果
        i = -1  # 最后一帧
        latest_kinematics = {
            'timestamp': timestamps[i],
            'x': track_history[i][1],
            'y': track_history[i][2],
            'yaw': track_history[i][3],
            'vx': vx_smooth[i],
            'vy': vy_smooth[i],
            'speed': np.sqrt(vx_smooth[i] ** 2 + vy_smooth[i] ** 2),
            'ax': ax_smooth[i],
            'ay': ay_smooth[i],
            'acceleration': np.sqrt(ax_smooth[i] ** 2 + ay_smooth[i] ** 2),
            'jerk': np.sqrt(jx_smooth[i] ** 2 + jy_smooth[i] ** 2),
        }

        return latest_kinematics
