import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
import numpy as np
from tools import config

class NuscenesDataset(Dataset):
    def __init__(self, nusc, scenes, config):
        self.nusc = nusc
        self.scenes = scenes
        self.config = config

        self.obs_len = config.OBS_LEN
        self.pred_len = config.PRED_LEN
        self.seq_len = self.obs_len + self.pred_len

        # 加载所有地图
        self.maps = {
            "singapore-onenorth": NuScenesMap(dataroot=config.DATAROOT, map_name="singapore-onenorth"),
            "singapore-hollandvillage": NuScenesMap(dataroot=config.DATAROOT, map_name="singapore-hollandvillage"),
            "singapore-queenstown": NuScenesMap(dataroot=config.DATAROOT, map_name="singapore-queenstown"),
            "boston-seaport": NuScenesMap(dataroot=config.DATAROOT, map_name="boston-seaport"),
        }

        self.sequences = self._create_sequences()

    def _get_traffic_light_status(self, sample_token, current_map):
        """
        获取当前sample中所有交通灯的状态。
        这是一个简化版本，实际应用中需要复杂的逻辑来关联agent和交通灯。
        这里我们只找场景中第一个被标注为红/黄/绿的灯。
        """
        sample = self.nusc.get('sample', sample_token)
        tl_status = [0, 0, 0, 1]  # [red, yellow, green, off] - 默认为off

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            if 'traffic_light' in ann['category_name']:
                # 获取交通灯状态属性
                attr_token = ann['attribute_tokens'][0]
                attr = self.nusc.get('attribute', attr_token)
                if 'red' in attr['name']:
                    return [1, 0, 0, 0]
                elif 'yellow' in attr['name']:
                    return [0, 1, 0, 0]
                elif 'green' in attr['name']:
                    return [0, 0, 1, 0]
        return tl_status  # 如果没有活动的灯，则返回off

    def _create_sequences(self):
        """为所有agent创建包含丰富特征的序列"""
        sequences = []
        print("Processing scenes to create enhanced sequences...")
        for scene in tqdm(self.scenes):
            log = self.nusc.get('log', scene['log_token'])
            location = log['location']
            current_map = self.maps[location]

            instance_trajs = {}

            first_sample_token = scene['first_sample_token']
            current_sample_token = first_sample_token

            while current_sample_token:
                sample = self.nusc.get('sample', current_sample_token)

                # 获取当前时间戳的交通灯状态
                tl_status_one_hot = self._get_traffic_light_status(current_sample_token, current_map)

                for ann_token in sample['anns']:
                    ann = self.nusc.get('sample_annotation', ann_token)
                    if 'vehicle' not in ann['category_name']:
                        continue

                    instance_token = ann['instance_token']
                    if instance_token not in instance_trajs:
                        instance_trajs[instance_token] = []

                    # 1. 位置信息
                    pos = np.array(ann['translation'][:2], dtype=np.float32)

                    # 2. 地图信息
                    is_on_drivable = float(current_map.get_is_on_layer(pos[0], pos[1], 'drivable_area'))

                    # 组合成一个时间步的特征
                    # [x, y, is_on_drivable, tl_red, tl_yellow, tl_green, tl_off]
                    step_features = np.concatenate([pos, [is_on_drivable], tl_status_one_hot])
                    instance_trajs[instance_token].append(step_features)

                if sample['next'] == '':
                    break
                current_sample_token = sample['next']

            # 处理每个agent的完整轨迹
            for token, traj in instance_trajs.items():
                if len(traj) < self.seq_len:
                    continue

                traj_array = np.array(traj, dtype=np.float32)

                # 3. 计算运动学特征 (速度和加速度)
                # traj_array[:, :2] 是位置信息
                pos_traj = traj_array[:, :2]
                vel_traj = np.diff(pos_traj, axis=0, prepend=pos_traj[0:1]) / 0.5  # NuScenes采样频率为2Hz (0.5s)
                acc_traj = np.diff(vel_traj, axis=0, prepend=vel_traj[0:1]) / 0.5

                # 组合所有特征: [x, y, vx, vy, ax, ay, is_drivable, tl_status...]
                full_feature_traj = np.hstack([pos_traj, vel_traj, acc_traj, traj_array[:, 2:]])

                # 使用滑动窗口切分
                for i in range(len(full_feature_traj) - self.seq_len + 1):
                    sequences.append(full_feature_traj[i: i + self.seq_len])

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        full_sequence = self.sequences[idx]

        # 划分观察和预测
        obs_full_traj = full_sequence[:self.obs_len]
        pred_pos_gt = full_sequence[self.obs_len:, :2]  # 目标只有(x,y)

        # 坐标相对化处理
        last_obs_pos = obs_full_traj[-1, :2]

        # 观察序列：位置相对化，其他特征保持不变
        obs_traj_rel = np.copy(obs_full_traj)
        obs_traj_rel[:, :2] -= last_obs_pos

        # 预测序列：位置相对化
        pred_traj_gt_rel = pred_pos_gt - last_obs_pos

        return {
            'obs_traj': torch.from_numpy(obs_traj_rel),
            'pred_traj_gt': torch.from_numpy(pred_traj_gt_rel)
        }