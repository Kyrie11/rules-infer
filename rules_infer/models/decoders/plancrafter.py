# -*- coding: utf-8 -*-
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class BehaviorEncoder(nn.Module):
    """将轨迹编码到行为特征空间（用于 FBD）。"""
    def __init__(self, d_out: int=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*30, 128), nn.ReLU(), nn.Linear(128, d_out))

    def forward(self, traj):  # traj: [B, T, 2]
        x = traj.reshape(traj.size(0), -1)
        return self.net(x)

def frechet_distance(mu_p, Sigma_p, mu_h, Sigma_h):
    # 伪实现：仅用均值差，协方差项留作 TODO
    return (mu_p - mu_h).pow(2).sum(-1).mean()

class PlanCrafter(nn.Module):
    """
    代价 = 安全/舒适/进度 + 风格/规范项(由 z_g 参数化)
    这里提供 human-like 评分计算接口（FBD + Norm Compliance）
    """
    def __init__(self):
        super().__init__()
        self.B = BehaviorEncoder(d_out=32)

    def forward(self, graph_feats, z_g, cig, init_state) -> Dict:
        # TODO: IL/MPC 外环求解器接口；此处返回占位规划轨迹
        B, T = init_state.shape[0], 30
        traj = torch.zeros(B, T, 2, device=init_state.device)
        return {"traj_ego": traj}

    def human_like_scores(self, traj_ego, human_bank, z_g_params) -> Dict:
        phi_p = self.B(traj_ego)
        mu_p, Sigma_p = phi_p.mean(0), torch.cov(phi_p.T)
        mu_h, Sigma_h = human_bank["mu"], human_bank["Sigma"]  # 预统计
        fbd = frechet_distance(mu_p, Sigma_p, mu_h, Sigma_h)
        # Norm-Compliance 示例：与 z_g 提供的礼让阈值/停止距离偏差
        norm_pen = torch.tensor(0.0, device=traj_ego.device)  # TODO
        return {"FBD": fbd, "NormPenalty": norm_pen}
