import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import os
import random
from tqdm import tqdm
import numpy as np

# 确保 nuscenes-devkit 已经安装
# pip install nuscenes-devkit
from nuscenes.nuscenes import NuScenes




# ----------------------------------
# 2. 数据集加载 (Dataset Loading)
# ----------------------------------
class NuScenesTrajectoryDataset(Dataset): # ... (代码与之前完全相同，为简洁省略)
    def __init__(self, nusc, config):
        self.nusc = nusc
        self.config = config
        self.maps = {m['filename']: NuScenesMap(dataroot=config['dataroot'], map_name=m['filename']) for m in nusc.map}
        self.sequences, self.full_trajectories = self._load_data()
    def _get_traffic_light_features(self, agent_pos, map_api):
        is_near_tl, dist_to_tl = 0.0, 1.0; tl_records = map_api.get_records_in_radius(agent_pos[0], agent_pos[1], 50, ['traffic_light']);
        if not tl_records: return np.array([is_near_tl, dist_to_tl], dtype=np.float32)
        min_dist = float('inf')
        for tl_id in tl_records:
            tl_polygon = map_api.get('traffic_light', tl_id);
            if not tl_polygon or 'polygon_token' not in tl_polygon: continue
            polygon = map_api.extract_polygon(tl_polygon['polygon_token']); center_point = np.mean(polygon.exterior.xy, axis=1); dist = np.linalg.norm(agent_pos - center_point);
            if dist < min_dist: min_dist = dist
        if min_dist < self.config['traffic_light_distance_threshold']: is_near_tl = 1.0
        dist_to_tl = min(min_dist, self.config['traffic_light_distance_threshold']) / self.config['traffic_light_distance_threshold']
        return np.array([is_near_tl, dist_to_tl], dtype=np.float32)
    def _load_data(self):
        all_sequences, full_trajectories = [], {}
        for scene in tqdm(self.nusc.scene, desc="Processing Scenes"):
            log = self.nusc.get('log', scene['log_token']); map_api = self.maps[log['location']]; first_sample_token = scene['first_sample_token']; sample = self.nusc.get('sample', first_sample_token)
            instance_tokens = {self.nusc.get('sample_annotation', ann_token)['instance_token'] for ann_token in sample['anns']}
            for instance_token in instance_tokens:
                trajectory = []; current_sample_token = first_sample_token
                while current_sample_token:
                    sample_record = self.nusc.get('sample', current_sample_token); instance_ann = None
                    for ann_token in sample_record['anns']:
                        ann = self.nusc.get('sample_annotation', ann_token)
                        if ann['instance_token'] == instance_token: instance_ann = ann; break
                    if instance_ann:
                        pos = instance_ann['translation']; agent_pos_2d = np.array([pos[0], pos[1]]); tl_features = self._get_traffic_light_features(agent_pos_2d, map_api); feature_vector = np.concatenate([agent_pos_2d, tl_features]); trajectory.append(feature_vector)
                    current_sample_token = sample_record['next']
                if len(trajectory) > 0: full_trajectories[(scene['token'], instance_token)] = np.array(trajectory, dtype=np.float32)
                total_len = self.config['history_len'] + self.config['future_len']
                if len(trajectory) >= total_len:
                    for i in range(len(trajectory) - total_len + 1):
                        hist = np.array(trajectory[i : i + self.config['history_len']], dtype=np.float32)
                        future = np.array([t[:2] for t in trajectory[i + self.config['history_len'] : i + total_len]], dtype=np.float32)
                        metadata = {"scene_token": scene['token'], "instance_token": instance_token, "start_index_in_full_traj": i, "full_traj_len": len(trajectory)}
                        all_sequences.append((hist, future, metadata))
        print(f"Finished processing. Found {len(all_sequences)} samples.")
        return all_sequences, full_trajectories
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        history, future, metadata = self.sequences[idx]
        return torch.from_numpy(history), torch.from_numpy(future), metadata
