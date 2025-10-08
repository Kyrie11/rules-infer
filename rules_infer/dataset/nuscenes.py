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
class NuScenesTrajectoryDataset(Dataset):
    """
    NuScenes anget 轨迹数据集
    """

    def __init__(self, nusc, config):
        self.nusc = nusc
        self.config = config
        self.sequences = self._load_sequences()

    def _load_sequences(self):
        """
        加载并处理所有场景中的所有 agent 轨迹
        """
        print("Loading and processing sequences from NuScenes...")
        all_sequences = []

        # 遍历每个场景
        for scene in tqdm(self.nusc.scene, desc="Processing Scenes"):
            first_sample_token = scene['first_sample_token']

            # 获取该场景中所有 agent 的 instance token
            sample = self.nusc.get('sample', first_sample_token)
            instance_tokens = {ann['instance_token'] for ann in sample['anns']}

            # 遍历每个 agent
            for instance_token in instance_tokens:
                trajectory = []
                current_sample_token = first_sample_token

                # 追踪这个 agent 在整个场景中的轨迹
                while current_sample_token:
                    sample_record = self.nusc.get('sample', current_sample_token)

                    # 查找当前 sample 中对应的 annotation
                    ann_tokens = sample_record['anns']
                    instance_ann = None
                    for ann_token in ann_tokens:
                        ann = self.nusc.get('sample_annotation', ann_token)
                        if ann['instance_token'] == instance_token:
                            instance_ann = ann
                            break

                    if instance_ann:
                        # 只关心 x, y 坐标
                        translation = instance_ann['translation']
                        trajectory.append([translation[0], translation[1]])

                    # 移动到下一个 sample
                    current_sample_token = sample_record['next']

                # 如果轨迹足够长，则使用滑动窗口创建样本
                total_len = self.config['history_len'] + self.config['future_len']
                if len(trajectory) >= total_len:
                    for i in range(len(trajectory) - total_len + 1):
                        hist = trajectory[i: i + self.config['history_len']]
                        future = trajectory[i + self.config['history_len']: i + total_len]
                        all_sequences.append((np.array(hist, dtype=np.float32), np.array(future, dtype=np.float32)))

        print(f"Finished processing. Found {len(all_sequences)} trajectory samples.")
        return all_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        history, future = self.sequences[idx]
        return torch.from_numpy(history), torch.from_numpy(future)
