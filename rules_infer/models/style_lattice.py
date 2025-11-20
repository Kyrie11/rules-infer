# -*- coding: utf-8 -*-
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
    std = (0.5*logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

class VectorQuantizerEMA(nn.Module):
    """
    VQ-EMA: Exponential Moving Average Vector Quantizer (D. van den Oord et al.)
    Args:
        num_codes: K
        dim: embedding dim
        decay: EMA decay
        eps: small const
    Returns:
        z_q, vq_loss (commitment term), perplexity (opt)
    """
    def __init__(self, num_codes: int, dim: int, decay: float = 0.99, eps: float = 1e-5, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.beta = beta

        embed = torch.randn(num_codes, dim)
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer('cluster_size', torch.zeros(num_codes))
        self.embed = nn.Parameter(embed)  # useful for init & fallback

    @torch.no_grad()
    def _update_ema(self, onehot: torch.Tensor, z_flat: torch.Tensor):
        # onehot: [N, K], z_flat: [N, d]
        n = onehot.sum(0)  # [K]
        embed_sum = onehot.t() @ z_flat  # [K, d]
        # EMA update
        self.cluster_size.mul_(self.decay).add_(n.cpu(), alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum.cpu(), alpha=1.0 - self.decay)
        # normalize
        n = self.cluster_size + self.eps
        new_embed = self.embed_avg / n.unsqueeze(1)
        # write to parameter (on CPU buffer then copy)
        self.embed.data.copy_(new_embed.to(self.embed.device))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [..., d]
        orig_shape = z.shape
        z_flat = z.view(-1, self.dim)  # [N, d]
        # distances
        # using (x - e)^2 = x^2 -2x e^T + e^2
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embed.t()
                + self.embed.pow(2).sum(1).unsqueeze(0))
        idx = dist.argmin(dim=1)  # [N]
        onehot = F.one_hot(idx, num_classes=self.num_codes).float()  # [N, K]
        z_q_flat = onehot @ self.embed  # [N, d]
        z_q = z_q_flat.view(*orig_shape)

        # commitment loss (straight-through)
        loss = F.mse_loss(z_q.detach(), z) * self.beta
        # straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # EMA update
        if self.training:
            # update buffers on CPU to avoid precision issues
            self._update_ema(onehot.detach(), z_flat.detach())

        # perplexity (optional)
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(- (avg_probs * (avg_probs + 1e-12).log()).sum())

        return z_q_st, loss, perplexity, idx.view(*z.shape[:-1])

# -------------------------
# Small Temporal Transformer for individuals
# -------------------------
class TemporalEncoder(nn.Module):
    """
    Simple transformer encoder along time dim.
    Input: [B*Na, T, d]
    Output: pooled [B*Na, d_out] (use mean or cls token)
    """
    def __init__(self, d_in: int, d_model: int = 128, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B*Na, T, d_in]
        h = self.input_proj(x)  # [B*Na, T, d_model]
        # mask: optional bool mask of shape [B*Na, T] where True means valid
        if mask is not None:
            # transformer expects src_key_padding_mask where True values are positions to be masked
            h = self.transformer(h, src_key_padding_mask=~mask)  # invert: mask True -> valid
        else:
            h = self.transformer(h)
        # pool over T
        h = h.transpose(1, 2)  # [B*Na, d_model, T]
        pooled = self.pool(h).squeeze(-1)  # [B*Na, d_model]
        return pooled

# -------------------------
# Individual Encoder (continuous + VQ)
# -------------------------
class IndividualEncoder(nn.Module):
    def __init__(self, feat_dim: int, time_dim: int, z_dim: int = 16, vq_k: int = 64, d_model: int = 128):
        """
        feat_dim: per-timestep feature dim (d)
        time_dim: T
        """
        super().__init__()
        self.temporal = TemporalEncoder(d_in=feat_dim, d_model=d_model, nlayers=2)
        self.fc = nn.Sequential(nn.Linear(self.temporal.out_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.mu = nn.Linear(d_model, z_dim)
        self.logvar = nn.Linear(d_model, z_dim)
        self.vq = VectorQuantizerEMA(vq_k, z_dim)

    def forward(self, H_A: torch.Tensor, a_mask: Optional[torch.Tensor] = None) -> Dict:
        # H_A: [B, Na, T, d]
        B, Na, T, d = H_A.shape
        x = H_A.view(B * Na, T, d)  # [B*Na, T, d]
        if a_mask is not None:
            # a_mask_time: [B, Na, T] or [B*Na, T] expected - assume a_mask_time not available -> ignore
            src_mask = None
        else:
            src_mask = None
        pooled = self.temporal(x, mask=None)  # [B*Na, d_model]
        h = self.fc(pooled)  # [B*Na, d_model]
        mu = self.mu(h)
        lv = self.logvar(h)
        z_cont = reparameterize(mu, lv)
        z_vq, vq_loss, perplexity, idx = self.vq(z_cont)
        # reshape back: [B, Na, z_dim]
        z_cont = z_cont.view(B, Na, -1)
        z_vq = z_vq.view(B, Na, -1)
        mu = mu.view(B, Na, -1)
        lv = lv.view(B, Na, -1)
        return {"z_cont": z_cont, "mu": mu, "logvar": lv, "z_vq": z_vq, "vq_loss": vq_loss, "perplexity": perplexity, "code_idx": idx.view(B, Na)}

# -------------------------
# Group Encoder (DeepSets + attention pooling)
# -------------------------
class AttentionPooling(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(), nn.Linear(d_hidden, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, d]
        scores = self.attn(x).squeeze(-1)  # [B, N]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)  # [B, N]
        pooled = (weights.unsqueeze(-1) * x).sum(1)  # [B, d]
        return pooled

class GroupEncoder(nn.Module):
    def __init__(self, zone_feat_dim: int, ctx_dim: int, z_dim: int = 16, vq_k: int = 32, hidden: int = 128):
        super().__init__()
        # per-agent -> zone aggregation is external (we assume zone_stats provided), but we allow extra per-agent pooling if needed
        self.set_fc = nn.Sequential(nn.Linear(zone_feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.pool = AttentionPooling(hidden)
        self.ctx_fc = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.vq = VectorQuantizerEMA(vq_k, z_dim)
        # prior from context
        self.prior_mu = nn.Linear(ctx_dim, z_dim)
        self.prior_logvar = nn.Linear(ctx_dim, z_dim)

    def forward(self, zone_stats: torch.Tensor, ctx: torch.Tensor, zone_mask: Optional[torch.Tensor] = None) -> Dict:
        # zone_stats: [B, Nz, zone_feat_dim]
        B, Nz, _ = zone_stats.shape
        h = self.set_fc(zone_stats)  # [B, Nz, hidden]
        agg = self.pool(h, mask=zone_mask)  # [B, hidden]
        ctx_h = self.ctx_fc(ctx)  # [B, hidden]
        fused = agg + ctx_h  # [B, hidden]
        mu = self.mu(fused)
        lv = self.logvar(fused)
        z_cont = reparameterize(mu, lv)
        z_vq, vq_loss, perplexity, idx = self.vq(z_cont.unsqueeze(1))  # reuse vq expecting [..., d]
        # vq returns shape [B,1,d] -> squeeze
        z_vq = z_vq.squeeze(1)
        # prior
        pmu = self.prior_mu(ctx)
        plv = self.prior_logvar(ctx)
        return {"z_cont": z_cont, "mu": mu, "logvar": lv, "z_vq": z_vq, "vq_loss": vq_loss, "perplexity": perplexity, "prior_mu": pmu, "prior_logvar": plv, "code_idx": idx.squeeze(1)}
# -------------------------
# Loss components: KL, InfoNCE, MMD (RBF)
# -------------------------
def kl_normal(mu: torch.Tensor, logvar: torch.Tensor, pmu: Optional[torch.Tensor] = None, plogvar: Optional[torch.Tensor] = None) -> torch.Tensor:
    if pmu is None or plogvar is None:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # conditional prior KL (batchwise mean)
    return 0.5 * torch.mean(plogvar - logvar + (logvar.exp() + (mu - pmu).pow(2)) / plogvar.exp() - 1)

def info_nce_loss(z_pos: torch.Tensor, z_anchor: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    z_pos: [N, d] positives (for anchor)
    z_anchor: [N, d] anchors
    Constructs per-row similarity vs all anchors in batch; positive is diagonal.
    Assumes matches are in same order.
    """
    z_pos_n = l2_normalize(z_pos, dim=-1)
    z_anchor_n = l2_normalize(z_anchor, dim=-1)
    logits = (z_anchor_n @ z_pos_n.t()) / temperature  # [N, N]
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def gaussian_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    # x: [n, d], y: [m, d]
    x2 = (x*x).sum(-1).unsqueeze(1)
    y2 = (y*y).sum(-1).unsqueeze(0)
    dist2 = x2 + y2 - 2 * (x @ y.t())
    return torch.exp(-dist2 / (2 * sigma * sigma))

def mmd_loss_rbf(x: torch.Tensor, y: torch.Tensor, sigmas=(1.0, 2.0, 4.0)) -> torch.Tensor:
    # unbiased MMD^2 estimate with multi-bandwidth
    Kxx = 0.0
    Kyy = 0.0
    Kxy = 0.0
    for s in sigmas:
        Kxx += gaussian_rbf(x, x, s)
        Kyy += gaussian_rbf(y, y, s)
        Kxy += gaussian_rbf(x, y, s)
    # subtract diag
    n = x.shape[0]
    m = y.shape[0]
    if n > 1:
        Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    else:
        Kxx = Kxx.sum() / (n * n + 1e-8)
    if m > 1:
        Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    else:
        Kyy = Kyy.sum() / (m * m + 1e-8)
    Kxy = Kxy.sum() / (n * m)
    mmd2 = Kxx + Kyy - 2 * Kxy
    return mmd2

# -------------------------
# Prototype memory for group prototypes
# -------------------------
class PrototypeMemory:
    """
    Keep EMA prototypes keyed by context (string/int).
    Example key can be tuple(time_of_day, weather, region) hashed to int.
    """
    def __init__(self, z_dim: int, decay: float = 0.99, device: Optional[torch.device] = None):
        self.decay = decay
        self.z_dim = z_dim
        self.device = device if device is not None else torch.device('cpu')
        self.store = {}  # key -> tensor

    def update(self, key, z: torch.Tensor):
        # z: [z_dim] or [B, z_dim] -> take mean
        if z.dim() == 2:
            z = z.mean(0)
        z = z.detach().to(self.device)
        if key in self.store:
            self.store[key] = self.store[key] * self.decay + (1.0 - self.decay) * z
        else:
            self.store[key] = z.clone()

    def get(self, key):
        return self.store.get(key, None)

def kl_normal(mu, logvar, pmu=None, plogvar=None):
    if pmu is None:  # 标准正态
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # 条件先验 KL
    return 0.5*torch.mean(plogvar - logvar + (logvar.exp() + (mu-pmu).pow(2))/plogvar.exp() - 1)

# -------------------------
# StyleLattice wrapper
# -------------------------
class StyleLattice(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 time_dim: int,
                 zone_feat_dim: int,
                 ctx_dim: int,
                 z_dim: int = 16,
                 K_ind: int = 64,
                 K_grp: int = 32,
                 beta: float = 2.0,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.Ei = IndividualEncoder(feat_dim, time_dim, z_dim=z_dim, vq_k=K_ind)
        self.Eg = GroupEncoder(zone_feat_dim, ctx_dim, z_dim=z_dim, vq_k=K_grp)
        self.beta = beta
        self.device = device if device else torch.device('cpu')
        self.prototype = PrototypeMemory(z_dim, decay=0.995, device=self.device)

    def forward(self, H_A: torch.Tensor, zone_stats: torch.Tensor, ctx: torch.Tensor,
                a_valid_mask: Optional[torch.Tensor] = None,
                zone_mask: Optional[torch.Tensor] = None,
                a2z_idx: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            H_A: [B, Na, T, d] - per-agent time features from ContextWeaver
            zone_stats: [B, Nz, s_dim]
            ctx: [B, ctx_dim]
            a_valid_mask: [B, Na] boolean (True valid)
            a2z_idx: [B, Na] int mapping agent->zone index
        Returns:
            dict with z_i, z_g, losses components
        """
        B, Na, T, d = H_A.shape
        # Individual encoding
        out_i = self.Ei(H_A, a_valid_mask)
        # Group encoding (per batch)
        out_g = self.Eg(zone_stats, ctx, zone_mask=zone_mask)

        # KL losses
        L_kl_i = kl_normal(out_i['mu'], out_i['logvar'])
        L_kl_g = kl_normal(out_g['mu'], out_g['logvar'], out_g['prior_mu'], out_g['prior_logvar'])

        # InfoNCE for group: we want z_g of same ctx (across batch) to be close
        # Build anchors and positives: use batch-level grouping by ctx (assume ctx identical exact values rare -> we use clustering by discrete ctx id in practice)
        # For demo: we compute InfoNCE between z_g_cont of items in batch and shuffled version with same ctx mask if such groups exist.
        z_g = out_g['z_cont']  # [B, z_dim]
        # For simplicity, use in-batch positives if ctx identical: we'll assume ctx provided as discrete keys in second half example
        # Here we implement a generic contrastive loss: pull together z_g[i] and z_g[j] if L2(ctx[i], ctx[j]) small.
        # We'll just use a simple scheme: treat each sample as anchor and its rotated version as positive (placeholder).
        # In practice, you should construct positive pairs by sampling different zones with identical c.
        info_nce = torch.tensor(0.0, device=z_g.device)
        # Placeholder: if B>1, treat pairs (i, j) where ctx equal (exact match)
        # We'll compare ctx via exact equality if ctx discrete; otherwise skip.
        try:
            # try exact match
            equal_ctx = (ctx.unsqueeze(1) == ctx.unsqueeze(0)).all(-1)  # [B,B]
            positives = []
            anchors = []
            for i in range(B):
                # find j != i with same ctx
                idxs = torch.where(equal_ctx[i].cpu())[0].to(z_g.device)
                idxs = idxs[idxs != i]
                if len(idxs) > 0:
                    j = idxs[0]
                    anchors.append(z_g[i])
                    positives.append(z_g[j])
            if len(anchors) > 0:
                anchors = torch.stack(anchors, dim=0)
                positives = torch.stack(positives, dim=0)
                info_nce = info_nce_loss(positives, anchors)
        except Exception:
            info_nce = torch.tensor(0.0, device=z_g.device)

        # Statistical matching (MMD) :
        # Idea: for each zone we may have empirical behavior statistics; we want generator conditioned on z_g to produce similar stats.
        # Here as placeholder we compute MMD between z_g and aggregated per-zone stats (requires a predictive head in M-3/4).
        # We'll compute MMD between group embedding distribution and some target distribution built from zone_stats mean
        target = zone_stats.mean(dim=1)  # [B, s_dim]
        # project both to same dim
        proj_t = nn.Linear(target.shape[-1], z_g.shape[-1]).to(z_g.device)
        tg = proj_t(target)  # [B, z_dim]
        mmd = mmd_loss_rbf(z_g, tg)

        # VQ losses
        vq_loss = out_i['vq_loss'].mean() + out_g['vq_loss'].mean()

        # total style loss
        loss_style = self.beta * (L_kl_i + L_kl_g) + vq_loss + info_nce + mmd

        # update prototype memory per ctx (use ctx as hashable key)
        # if ctx is discrete/int tensor, use that; otherwise use bytes
        for b in range(B):
            key = None
            if ctx.shape[-1] == 1 and ctx.dtype in (torch.int64, torch.int32):
                key = int(ctx[b].item())
            else:
                key = tuple((ctx[b].detach().cpu().numpy()).tolist())
            self.prototype.update(key, out_g['z_cont'][b])

        return {
            "z_i": out_i['z_vq'],                 # [B,Na,z]
            "z_i_cont": out_i['z_cont'],
            "z_g": out_g['z_vq'],                 # [B, z]
            "z_g_cont": out_g['z_cont'],
            "loss_style": loss_style,
            "loss_components": {
                "kl_i": L_kl_i, "kl_g": L_kl_g, "vq": vq_loss, "info_nce": info_nce, "mmd": mmd
            },
            "aux": {"ind_perplex": out_i['perplexity'], "grp_perplex": out_g['perplexity']}
        }

class ZoneStatHead(nn.Module):
    """让 H_Z 直接输出 zone 统计（学生路）。"""
    def __init__(self, d_in: int, s_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, s_dim)
        )
    def forward(self, H_Z):  # [B, Nz, d]
        return self.net(H_Z) # [B, Nz, s_dim]

class ZoneStatTeacher(nn.Module):
    """
    教师路：从 H_A 聚合到 zone 统计（训练时构造弱目标，stop_grad 到 H_A）。
    不需要解码框，只用 a2z_idx 和 mask 做可微的统计近似。
    """
    def __init__(self, d_a: int, s_dim: int):
        super().__init__()
        # 把 agent 特征压成统计充分量，再按 zone 聚合
        self.feat2mom = nn.Sequential(
            nn.Linear(d_a, 128), nn.ReLU(),
            nn.Linear(128, s_dim)  # 每个 agent 产生对 zone 统计的“贡献”
        )

    def forward(self, H_A, a2z_idx, a_valid_mask, Nz):
        """
        H_A: [B, Na, T, d] -> 先做时序池化
        a2z_idx: [B, Na] in [0..Nz-1] or -1
        """
        B, Na, T, d = H_A.shape
        A_pool = H_A.mean(dim=2)  # [B, Na, d] 简单时间均值（可换成 TemporalEncoder）
        contrib = self.feat2mom(A_pool)  # [B, Na, s_dim]
        # 聚合到每个 zone：sum/mean(对应该 zone 的 agents)
        s_dim = contrib.size(-1)
        Zt = H_A.new_zeros(B, Nz, s_dim)
        cnt = H_A.new_zeros(B, Nz, 1)
        for b in range(B):
            z_index = a2z_idx[b]   # [Na]
            mask = (z_index >= 0) & a_valid_mask[b].bool()
            if mask.any():
                z = z_index[mask]                  # 这些 agent 的 zone 索引
                add = contrib[b][mask]             # [n, s_dim]
                Zt[b].index_add_(0, z, add)        # 按 zone 求和
                one = torch.ones(z.shape[0], 1, device=H_A.device, dtype=Zt.dtype)
                cnt[b].index_add_(0, z, one)
        Zt = Zt / (cnt.clamp_min(1.0))
        return Zt.detach()  # stop-grad: 作为学生监督目标

