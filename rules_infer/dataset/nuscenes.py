# dataset.py (Updated)

import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
import numpy as np
from tqdm import tqdm


class NuScenesTrajectoryDataset(Dataset):
    def __init__(self, nusc, hist_len=8, pred_len=12, split='v1.0-trainval'):
        self.nusc = nusc
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.seq_len = hist_len + pred_len
        self.samples = []
        self._prepare_data(split)

    def _prepare_data(self, split):
        print("Preparing data samples (including vehicles and humans)...")
        scene_splits = self._get_scene_splits(split)

        for scene in tqdm(self.nusc.scene):
            if scene['name'] not in scene_splits:
                continue

            first_sample_token = scene['first_sample_token']
            sample_tokens = self._get_sample_tokens_in_scene(first_sample_token)
            instance_trajectories = self._get_instance_trajectories(sample_tokens)

            for instance_token, trajectory in instance_trajectories.items():
                if len(trajectory) >= self.seq_len:
                    for i in range(len(trajectory) - self.seq_len + 1):
                        self.samples.append(trajectory[i: i + self.seq_len])

    def _get_scene_splits(self, split_name):
        from nuscenes.utils.splits import create_splits_scenes
        splits = create_splits_scenes()
        return splits[split_name]

    def _get_sample_tokens_in_scene(self, first_token):
        tokens = []
        current_token = first_token
        while current_token != "":
            tokens.append(current_token)
            sample = self.nusc.get('sample', current_token)
            current_token = sample['next']
        return tokens

    def _get_instance_trajectories(self, sample_tokens):
        trajectories = {}
        for token in sample_tokens:
            sample = self.nusc.get('sample', token)
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)

                # --- MODIFICATION START ---
                # 检查类别是否为车辆或行人
                is_vehicle = ann['category_name'].startswith('vehicle')
                is_human = ann['category_name'].startswith('human')

                if is_vehicle or is_human:
                    # --- MODIFICATION END ---
                    instance_token = ann['instance_token']
                    if instance_token not in trajectories:
                        trajectories[instance_token] = []
                    trajectories[instance_token].append(ann['translation'][:2])  # (x, y)
        return trajectories

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_trajectory = np.array(self.samples[idx], dtype=np.float32)
        history = full_trajectory[:self.hist_len]
        future = full_trajectory[self.hist_len:]
        origin = history[-1].copy()
        history -= origin
        future -= origin

        return torch.from_numpy(history), torch.from_numpy(future)
