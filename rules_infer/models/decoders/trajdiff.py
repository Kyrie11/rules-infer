# -*- coding: utf-8 -*-
from typing import Dict
import torch
import torch.nn as nn

class TrajUNet1D(nn.Module):
    """简化 1D UNet（点序列），可替换为更强架构。"""
    def __init__(self, d: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, d))

    def forward(self, x, cond):
        # x: [B, Na, T, 2] 噪声轨迹; cond: [B, Na, d_cond]
        # TODO: 组装条件（H/z_i/z_g/pi/intent），time embedding, classifier-free guidance
        h = self.net(cond)
        # 伪输出：同形态占位
        return torch.zeros_like(x)

class TrajDiff(nn.Module):
    """
    输出: 每个 agent 的 M 条轨迹 + 协方差
    训练: 标准 DDPM/Flow-Matching 损失（此处留接口）
    """
    def __init__(self, T: int=30, M: int=8, d_cond: int=256):
        super().__init__()
        self.M, self.T = M, T
        self.model = TrajUNet1D(d=d_cond)

    def forward(self, cond: Dict) -> Dict:
        # 伪采样：返回占位的多模态轨迹
        B, Na = cond["H_A"].shape[:2]
        trajs = torch.zeros(B, Na, self.M, self.T, 2, device=cond["H_A"].device)
        covs  = torch.eye(2, device=trajs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B,Na,self.M,1,1)
        return {"trajs": trajs, "covs": covs}

    def training_step(self, gt_trajs, cond):
        """
        gt_trajs: [B, Na, T, 2]
        cond: Dict embedding
        """
        # 1. 采样时刻 t
        B, Na, T, _ = gt_trajs.shape
        t = torch.randint(0, 1000, (B,), device=gt_trajs.device).long()

        # 2. 加噪 (Forward Diffusion)
        noise = torch.randn_like(gt_trajs)
        # 假设 self.scheduler 提供了 alpha_cumprod
        # alpha_bar = self.scheduler.alphas_cumprod[t]
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        # 简化伪写：
        x_t = gt_trajs * 0.5 + noise * 0.5

        # 3. 预测噪声 (Model Forward)
        # 需将 cond 展平并与 t 结合注入 UNet
        # 这里假设 net 输出 noise prediction
        pred_noise = self.model(x_t, cond, t)

        # 4. 损失 (Simple MSE)
        loss = F.mse_loss(pred_noise, noise)
        return loss