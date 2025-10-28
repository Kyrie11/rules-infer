import torch
from torch.utils.data import Dataset, DataLoader, random_split
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
import numpy as np
from nuscenes.nuscenes import NuScenes




# ----------------------------------
# 2. 数据集加载 (Dataset Loading)
# ----------------------------------
class NuScenesTrajectoryDataset(Dataset):
    def __init__(self, nusc, config):
        self.nusc = nusc
        self.config = config
        print("Loading NuScenes maps...")
        self.maps = {
            location: NuScenesMap(dataroot=self.config['dataroot'], map_name=location)
            for location in [
                'singapore-hollandvillage', 'singapore-queenstown',
                'boston-seaport', 'singapore-onenorth'
            ]
        }
        print("Maps loaded successfully.")
        self.sequences = self._load_data()

    def _get_traffic_light_features(self, agent_pos, map_api):
        is_near_tl, dist_to_tl = 0.0, 1.0
        records = map_api.get_records_in_radius(agent_pos[0], agent_pos[1], 50, ['traffic_light'])
        tl_tokens = records.get('traffic_light')
        if not tl_tokens:
            return np.array([is_near_tl, dist_to_tl], dtype=np.float32)

        min_dist = float('inf')
        for tl_token in tl_tokens:
            tl_record = map_api.get('traffic_light', tl_token)
            if not tl_record or 'polygon_token' not in tl_record: continue
            polygon = map_api.extract_polygon(tl_record['polygon_token'])
            center_point = np.mean(polygon.exterior.xy, axis=1)
            dist = np.linalg.norm(agent_pos - center_point)
            if dist < min_dist: min_dist = dist

        if min_dist < self.config['traffic_light_distance_threshold']:
            is_near_tl = 1.0
        dist_to_tl = min(min_dist, self.config['traffic_light_distance_threshold']) / self.config[
            'traffic_light_distance_threshold']
        return np.array([is_near_tl, dist_to_tl], dtype=np.float32)

    def _load_data(self):
        all_sequences = []
        for scene in tqdm(self.nusc.scene, desc="Processing Scenes for Dataset"):
            log_record = self.nusc.get('log', scene['log_token'])
            location = log_record['location']
            map_api = self.maps[location]

            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                for ann_token in sample['anns']:
                    ann = self.nusc.get('sample_annotation', ann_token)
                    # 只处理车辆
                    if 'vehicle' not in ann['category_name']: continue

                    # 检查是否有足够的历史和未来
                    has_history = ann['prev'] != ''
                    has_future = ann['next'] != ''
                    if not (has_history and has_future): continue

                    # 提取完整的轨迹片段
                    total_len = self.config['history_len'] + self.config['future_len']
                    trajectory = []

                    # 提取历史
                    current_ann = ann
                    for _ in range(self.config['history_len']):
                        pos = current_ann['translation']
                        agent_pos_2d = np.array([pos[0], pos[1]])
                        tl_features = self._get_traffic_light_features(agent_pos_2d, map_api)
                        feature_vector = np.concatenate([agent_pos_2d, tl_features])
                        trajectory.insert(0, feature_vector)
                        if current_ann['prev'] == '': break
                        current_ann = self.nusc.get('sample_annotation', current_ann['prev'])

                    # 提取未来
                    current_ann = ann
                    for _ in range(self.config['future_len']):
                        if current_ann['next'] == '': break
                        current_ann = self.nusc.get('sample_annotation', current_ann['next'])
                        pos = current_ann['translation']
                        agent_pos_2d = np.array([pos[0], pos[1]])
                        tl_features = self._get_traffic_light_features(agent_pos_2d, map_api)
                        feature_vector = np.concatenate([agent_pos_2d, tl_features])
                        trajectory.append(feature_vector)

                    if len(trajectory) == total_len:
                        hist = np.array(trajectory[:self.config['history_len']], dtype=np.float32)
                        future = np.array([t[:2] for t in trajectory[self.config['history_len']:]], dtype=np.float32)
                        all_sequences.append((hist, future))

                sample_token = sample['next']

        print(f"Finished processing. Found {len(all_sequences)} samples.")
        return all_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        history, future = self.sequences[idx]
        return torch.from_numpy(history), torch.from_numpy(future)

