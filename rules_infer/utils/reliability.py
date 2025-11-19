# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    """logits 温度缩放校准。"""
    def __init__(self, T_init: float=1.0):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(T_init))

    def forward(self, logits):
        return logits / self.T.clamp(min=1e-3)

def safety_filter(trajs, map_polys, agents):
    """
    运行时安全约束：碰撞/越界/TTC 过滤或代价抬升（占位函数）。
    """
    # TODO: 实现几何检测并过滤/惩罚
    return trajs
