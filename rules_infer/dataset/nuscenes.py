# dataset.py (Updated)

import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box


class NuScenesTrajectoryDataset(Dataset):
    def __init__(self, config, nusc):
        self.config = config
        self.nusc = nusc
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        seq_len = self.config.HIST_LEN + self.config.PRED_LEN

        for scene in tqdm(self.nusc.scene, desc="Processing Scenes for Dataset"):
            instance_tracks = {}
            first_sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', first_sample_token)

            # 遍历场景中的所有sample
            while sample:
                for ann_token in sample['anns']:
                    ann = self.nusc.get('sample_annotation', ann_token)
                    # 只关心车辆
                    if 'vehicle' in ann['category_name']:
                        instance_token = ann['instance_token']
                        if instance_token not in instance_tracks:
                            instance_tracks[instance_token] = []

                        # 获取车辆在全局坐标系下的信息
                        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                        pos = box.center[:2]  # 全局 x, y

                        instance_tracks[instance_token].append({
                            'pos': pos,
                            'sample_token': sample['token'],
                            'instance_token': instance_token,
                            'timestamp': sample['timestamp']
                        })

                if not sample['next']:
                    break
                sample = self.nusc.get('sample', sample['next'])

            # 从每个instance的完整轨迹中切分出历史和未来序列
            for instance_token, track in instance_tracks.items():
                if len(track) >= seq_len:
                    for i in range(len(track) - seq_len + 1):
                        sequence_data = track[i: i + seq_len]

                        # 将坐标相对于历史轨迹的最后一个点进行归一化
                        origin = sequence_data[self.config.HIST_LEN - 1]['pos']

                        history = np.array([p['pos'] - origin for p in sequence_data[:self.config.HIST_LEN]])
                        future = np.array([p['pos'] - origin for p in sequence_data[self.config.HIST_LEN:]])

                        sequences.append((
                            torch.tensor(history, dtype=torch.float32),
                            torch.tensor(future, dtype=torch.float32)
                        ))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
