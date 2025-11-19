# -*- coding: utf-8 -*-
import torch

def soft_and(a, b):  # Łukasiewicz t-norm 简版
    return torch.clamp(a + b - 1.0, min=0.0, max=1.0)

def soft_imply(a, b):  # a => b  等价于 min(1, 1-a + b)
    return torch.clamp(1.0 - a + b, 0.0, 1.0)

def satisfaction_yield_to(thw: torch.Tensor, tau: float=1.5):
    """THW >= tau 的满足度映射到 [0,1]。"""
    return torch.clamp((thw - tau) / tau + 1.0, 0.0, 1.0)  # 线性示意

def rule_penalty(sat: torch.Tensor):
    return 1.0 - sat.mean()


def compute_thw(traj_ego, traj_target, dt=0.1):
    """
    可微 THW 计算。
    traj: [B, T, 2]
    """
    # 简化：计算两车距离除以 ego 速度
    dist = torch.norm(traj_ego - traj_target, dim=-1)  # [B, T]

    # 速度估算
    vel = torch.norm(traj_ego[:, 1:] - traj_ego[:, :-1], dim=-1) / dt  # [B, T-1]
    vel = F.pad(vel, (0, 1), "replicate")

    # 避免除零
    thw = dist / (vel + 1e-5)
    return thw.min(dim=-1).values  # 取全轨迹最小 THW

def satisfaction_yield_to(traj_ego, traj_target, tau=2.0):
    min_thw = compute_thw(traj_ego, traj_target)
    # 逻辑：THW >= tau -> 满足度 1.0
    # 使用 Sigmoid 或分段线性作为 Soft Logic 谓词
    # slope 控制硬度
    return torch.sigmoid(5.0 * (min_thw - tau))
