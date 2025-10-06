import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
import numpy as np


# 注意：你可能需要 pip install pyquaternion
# from pyquaternion import Quaternion # 这个库在这个实现中不是必需的，但处理旋转时常用

class NuscenesDataset(Dataset):
    def __init__(self, nusc, scenes, config):
        self.nusc = nusc
        self.scenes = scenes
        self.config = config

        self.obs_len = config.OBS_LEN
        self.pred_len = config.PRED_LEN
        self.seq_len = self.obs_len + self.pred_len

        # **[修改点 1]** 使用懒加载（Lazy Loading）策略
        self.maps = {}

        # 定义前向搜索交通灯的距离阈值
        self.SEARCH_DIST_AHEAD_M = 50.0

        self.sequences = self._create_sequences()

    def _get_map(self, location):
        """按需加载地图，并缓存起来"""
        if location not in self.maps:
            print(f"Loading map for location: {location}...")
            self.maps[location] = NuScenesMap(dataroot=self.config.DATAROOT, map_name=location)
        return self.maps[location]

    # **[修改点 2]** 实现了新的、与agent关联的交通灯状态获取方法
    def _get_agent_specific_traffic_light_status(self, ann, sample, current_map):
        """
        获取与特定agent前行路径相关的交通灯状态。
        """
        tl_status = [0, 0, 0, 1]  # [red, yellow, green, off]
        agent_pos = np.array(ann['translation'][:2])

        try:
            lane_token = current_map.get_closest_lane(agent_pos[0], agent_pos[1], radius=2.0)
            if not lane_token:
                return tl_status
        except Exception:
            return tl_status

        lanes_to_visit = [(lane_token, 0.0)]
        visited_lanes = set()
        relevant_tl_token = None

        while lanes_to_visit:
            current_lane_token, dist = lanes_to_visit.pop(0)

            if current_lane_token in visited_lanes:
                continue
            visited_lanes.add(current_lane_token)

            # ==================== 主要修改点在这里 ====================
            lane_record = current_map.get('lane', current_lane_token)

            # 安全地访问 'traffic_control_tokens'
            traffic_control_tokens = lane_record.get('traffic_control_tokens', [])

            if traffic_control_tokens:
                # 检查这些控制器中是否有交通灯
                for tc_token in traffic_control_tokens:
                    tc_record = current_map.get('traffic_control', tc_token)
                    # 确保记录存在且类型是 'traffic_light'
                    if tc_record and tc_record.get('kind') == 'traffic_light':
                        relevant_tl_token = tc_token
                        break  # 找到了一个交通灯，停止检查这个车道的其他控制器

                if relevant_tl_token:
                    break  # 找到了我们关心的交通灯，停止整个前向搜索
            # ========================================================

            if dist > self.SEARCH_DIST_AHEAD_M:
                continue

            outgoing_lane_tokens = current_map.get_outgoing_lane_ids(current_lane_token)
            if outgoing_lane_tokens:
                new_dist = dist + 10.0
                for next_lane_token in outgoing_lane_tokens:
                    if next_lane_token not in visited_lanes:
                        lanes_to_visit.append((next_lane_token, new_dist))

        if relevant_tl_token:
            for ann_token in sample['anns']:
                ann_record = self.nusc.get('sample_annotation', ann_token)
                if ann_record['instance_token'] == relevant_tl_token:
                    # 确保有 attribute_tokens
                    if not ann_record['attribute_tokens']:
                        continue
                    attr_token = ann_record['attribute_tokens'][0]
                    attr = self.nusc.get('attribute', attr_token)
                    if 'red' in attr['name']:
                        return [1, 0, 0, 0]
                    elif 'yellow' in attr['name']:
                        return [0, 1, 0, 0]
                    elif 'green' in attr['name']:
                        return [0, 0, 1, 0]
                    break
        return tl_status

    def _create_sequences(self):
        """为所有agent创建包含丰富特征的序列"""
        sequences = []
        print("Processing scenes to create enhanced sequences...")
        for scene in tqdm(self.scenes):
            log = self.nusc.get('log', scene['log_token'])
            location = log['location']
            current_map = self._get_map(location)

            instance_trajs = {}

            first_sample_token = scene['first_sample_token']
            current_sample_token = first_sample_token

            while current_sample_token:
                sample = self.nusc.get('sample', current_sample_token)

                for ann_token in sample['anns']:
                    ann = self.nusc.get('sample_annotation', ann_token)
                    if 'vehicle' not in ann['category_name']:
                        continue

                    instance_token = ann['instance_token']
                    if instance_token not in instance_trajs:
                        instance_trajs[instance_token] = []

                    # 1. 位置信息
                    pos = np.array(ann['translation'][:2], dtype=np.float32)
                    patch_box = (pos[0] - 0.5, pos[1] - 0.5, pos[0] + 0.5, pos[1] + 0.5)
                    # 查询这个patch内是否有 'drivable_area' 的记录
                    records_in_patch = current_map.get_records_in_patch(patch_box, layer_names=['drivable_area'])

                    # 如果返回的字典中 'drivable_area' 列表不为空，则说明agent在可行驶区域内
                    is_on_drivable = float(len(records_in_patch['drivable_area']) > 0)


                    # **[修改点 3]** 调用新的函数获取agent专属的交通灯状态
                    tl_status_one_hot = self._get_agent_specific_traffic_light_status(ann, sample, current_map)

                    # 组合成一个时间步的特征
                    # [x, y, is_on_drivable, tl_red, tl_yellow, tl_green, tl_off]
                    step_features = np.concatenate([pos, [is_on_drivable], tl_status_one_hot])
                    instance_trajs[instance_token].append(step_features)

                if sample['next'] == '':
                    break
                current_sample_token = sample['next']

            # 处理每个agent的完整轨迹 (这部分逻辑与你原来的一样，是正确的)
            for token, traj in instance_trajs.items():
                if len(traj) < self.seq_len:
                    continue

                traj_array = np.array(traj, dtype=np.float32)

                # 3. 计算运动学特征 (速度和加速度)
                pos_traj = traj_array[:, :2]
                vel_traj = np.diff(pos_traj, axis=0, prepend=pos_traj[0:1]) / 0.5
                acc_traj = np.diff(vel_traj, axis=0, prepend=vel_traj[0:1]) / 0.5

                # 组合所有特征: [x, y, vx, vy, ax, ay, is_drivable, tl_status...]
                full_feature_traj = np.hstack([pos_traj, vel_traj, acc_traj, traj_array[:, 2:]])

                for i in range(len(full_feature_traj) - self.seq_len + 1):
                    sequences.append(full_feature_traj[i: i + self.seq_len])

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        full_sequence = self.sequences[idx]
        obs_full_traj = full_sequence[:self.obs_len]
        pred_pos_gt = full_sequence[self.obs_len:, :2]
        last_obs_pos = obs_full_traj[-1, :2]
        obs_traj_rel = np.copy(obs_full_traj)
        obs_traj_rel[:, :2] -= last_obs_pos
        pred_traj_gt_rel = pred_pos_gt - last_obs_pos

        return {
            'obs_traj': torch.from_numpy(obs_traj_rel),
            'pred_traj_gt': torch.from_numpy(pred_traj_gt_rel)
        }

