# -*- coding: utf-8 -*-
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterize(mu, logvar):
    std = (0.5*logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

class VectorQuantizer(nn.Module):
    """简化版 VQ (EMA 版请自行替换)。"""
    def __init__(self, num_codes: int, d: int, beta_commit: float = 0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, d))
        self.beta = beta_commit

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z:[..., d] -> 找最近code
        z_flat = z.view(-1, z.shape[-1])
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2*z_flat @ self.codebook.t()
                + self.codebook.pow(2).sum(1, keepdim=True).t())
        idx = dist.argmin(dim=1)
        z_q = self.codebook[idx].view_as(z)
        # commit & codebook loss
        loss_commit = F.mse_loss(z_q.detach(), z)
        loss_code   = F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()  # straight-through
        return z_q, self.beta*(loss_commit + loss_code)

class IndividualEncoder(nn.Module):
    """个体风格编码器：输入为统计+时序特征，输出 μ_i, logσ_i 与离散code。"""
    def __init__(self, d_in: int = 64, d_z: int = 16, K: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.mu = nn.Linear(128, d_z)
        self.lv = nn.Linear(128, d_z)
        self.vq = VectorQuantizer(K, d_z)

    def forward(self, feats: torch.Tensor) -> Dict:
        h = self.backbone(feats)
        mu, lv = self.mu(h), self.lv(h)
        z_cont = reparameterize(mu, lv)
        z_q, vq_loss = self.vq(z_cont)
        return {"z_cont": z_cont, "mu": mu, "logvar": lv, "z_vq": z_q, "vq_loss": vq_loss}

class GroupEncoder(nn.Module):
    """群体风格编码器：Zone 内聚合统计 + 上下文 c。"""
    def __init__(self, d_in: int = 64, d_ctx: int = 32, d_z: int = 16, K: int = 32):
        super().__init__()
        self.set_fc = nn.Sequential(nn.Linear(d_in,128), nn.ReLU(), nn.Linear(128,128))
        self.ctx_fc = nn.Linear(d_ctx, 128)
        self.mu  = nn.Linear(128, d_z)
        self.lv  = nn.Linear(128, d_z)
        self.vq  = VectorQuantizer(K, d_z)

        self.prior_mu = nn.Linear(d_ctx, d_z)     # p(z_g|c)
        self.prior_lv = nn.Linear(d_ctx, d_z)

    def forward(self, Z_stats: torch.Tensor, ctx: torch.Tensor) -> Dict:
        # Z_stats: [B, Nz, d_in] 已用 DeepSets 聚合; ctx:[B,d_ctx]
        h = self.set_fc(Z_stats).mean(1) + self.ctx_fc(ctx)
        mu, lv = self.mu(h), self.lv(h)
        z_cont = reparameterize(mu, lv)
        z_q, vq_loss = self.vq(z_cont)
        # 条件先验
        pmu, plv = self.prior_mu(ctx), self.prior_lv(ctx)
        return {"z_cont": z_cont, "mu": mu, "logvar": lv,
                "z_vq": z_q, "vq_loss": vq_loss,
                "prior_mu": pmu, "prior_logvar": plv}

def kl_normal(mu, logvar, pmu=None, plogvar=None):
    if pmu is None:  # 标准正态
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # 条件先验 KL
    return 0.5*torch.mean(plogvar - logvar + (logvar.exp() + (mu-pmu).pow(2))/plogvar.exp() - 1)

class StyleLattice(nn.Module):
    """封装：个体/群体风格推断与损失计算（KL/VQ/对比/统计匹配占位）。"""
    def __init__(self, d_ind_in=64, d_grp_in=64, d_ctx=32, z_dim=16, K_ind=64, K_grp=32, beta=2.0):
        super().__init__()
        self.Ei = IndividualEncoder(d_ind_in, z_dim, K_ind)
        self.Eg = GroupEncoder(d_grp_in, d_ctx, z_dim, K_grp)
        self.beta = beta

    def forward(self, ind_feats, grp_feats, ctx) -> Dict:
        out_i = self.Ei(ind_feats)
        out_g = self.Eg(grp_feats, ctx)
        # KL losses
        L_kl_i = kl_normal(out_i["mu"], out_i["logvar"])
        L_kl_g = kl_normal(out_g["mu"], out_g["logvar"], out_g["prior_mu"], out_g["prior_logvar"])
        # TODO: InfoNCE(同 context 拉近 z_g)、统计匹配(MMD/KL) —— 留作实现
        return {
            "z_i": out_i["z_vq"], "z_i_cont": out_i["z_cont"],
            "z_g": out_g["z_vq"], "z_g_cont": out_g["z_cont"],
            "loss_style": self.beta*(L_kl_i+L_kl_g) + out_i["vq_loss"] + out_g["vq_loss"],
            "aux": {"kl_i": L_kl_i, "kl_g": L_kl_g}
        }
