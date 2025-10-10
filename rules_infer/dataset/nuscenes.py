import torch
from torch.utils.data import Dataset, DataLoader, random_split
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
import numpy as np
from nuscenes.nuscenes import NuScenes




# ----------------------------------
# 2. 数据集加载 (Dataset Loading)
# ----------------------------------
class NuScenesTrajectoryDataset(Dataset): # ... (代码与之前完全相同，为简洁省略)
    # 在 NuScenesTrajectoryDataset 类中
    def __init__(self, nusc, config):
        self.nusc = nusc
        self.config = config
        print("Loading NuScenes maps using location names...")

        ### MODIFICATION START ###
        # 创建一个从 log_token 到 location 的映射
        log_token_to_location = {log['token']: log['location'] for log in self.nusc.log}

        self.maps = {}
        # 遍历 nuscenes 的地图记录
        for map_record in self.nusc.map:
            # 地图记录中包含了它所关联的所有 log_tokens
            log_tokens = map_record['log_tokens']

            # 假设同一张地图文件（如 boston-seaport）的所有 log 都共享同一个 location
            # 我们取第一个 log_token 来查找 location
            if not log_tokens:
                continue

            first_log_token = log_tokens[0]
            location = log_token_to_location.get(first_log_token)

            if location is None:
                print(f"Warning: Could not find location for map record: {map_record['token']}")
                continue

            # 检查 location (即 map_name) 是否已经加载
            if location not in self.maps:
                # 使用 location 作为 map_name 来加载地图
                # NuScenesMap 内部会自动处理文件路径
                try:
                    print(f"  - Loading map for location: {location}")
                    self.maps[location] = NuScenesMap(
                        dataroot=self.config['dataroot'],
                        map_name=location
                    )
                except Exception as e:
                    print(f"  - Failed to load map {location}: {e}")

        ### MODIFICATION END ###

        if not self.maps:
            raise RuntimeError("Failed to load any maps. Please check your NuScenes installation and dataroot.")

        print("Maps loaded successfully.")

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
