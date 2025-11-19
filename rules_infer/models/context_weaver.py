from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class PosEnc(nn.Module):
    """简易时间位置编码，可替换为更强的旋转位置编码。"""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(4, d_model)  # [t_norm, sin, cos, phase(optional)]

    def forward(self, t_feat: torch.Tensor) -> torch.Tensor:
        return self.proj(t_feat)

class AgentEncoder(nn.Module):
    """
    输入: [B, N_a, T, F_agent]  (位置/速度/加速度/航向/可见性...)
    输出: [B, N_a, d]  (per-agent token) 及可选时序 token [B, N_a, T, d]
    """
    def __init__(self, d: int = 192, n_layers: int = 3, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(16, d)  # 16: 你可按实际特征维度替换
        encoder_layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4*d, batch_first=True)
        self.temporal = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, T, F = x.shape
        x = self.input_proj(x)                  # [B, N, T, d]
        x = x.view(B*N, T, -1)
        h = self.temporal(x, src_key_padding_mask=mask)  # [B*N, T, d]
        h_last = h[:, -1]                       # [B*N, d]
        return h_last.view(B, N, -1), h.view(B, N, T, -1)

class PolylineEncoder(nn.Module):
    """
    对 lanelet/zone 多段线进行向量化编码。
    输入: [B, N_poly, P, 2/3 + attrs]
    输出: [B, N_poly, d]
    """
    def __init__(self, d: int = 192):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, d), nn.ReLU(), nn.Linear(d, d))
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, poly_feats: torch.Tensor) -> torch.Tensor:
        # 伪实现: 点级投影+max-pool
        B, N, P, F = poly_feats.shape
        x = self.mlp(poly_feats)                # [B, N, P, d]
        x = x.transpose(2, 3)                   # [B, N, d, P]
        x = self.pool(x).squeeze(-1)            # [B, N, d]
        return x


class CrossFusion(nn.Module):
    """
    Agent ↔ Lane/Zone 融合 (半径/拓扑裁剪后的邻域).
    """
    def __init__(self, d: int = 192, n_heads: int = 4):
        super().__init__()
        self.cross_attn_AL = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.cross_attn_AZ = nn.MultiheadAttention(d, n_heads, batch_first=True)

    def forward(self, H_A: torch.Tensor, H_L: torch.Tensor, H_Z: torch.Tensor,
                mask_AL: torch.Tensor, mask_AZ: torch.Tensor) -> torch.Tensor:
        """
        idx_AL/idx_AZ: 邻域索引（每个 agent 对应可见的 lane/zone 列表）。为便于示例，这里伪实现成全连接。
        """
        # 实际实现：为每个 agent 收集邻域L/Z tokens，拼batch，做 MHA 后scatter回去。
        H_A2L, _ = self.cross_attn_AL(H_A, H_L, H_L, attn_mask=mask_AL)  # 伪全连接
        H_A2Z, _ = self.cross_attn_AZ(H_A, H_Z, H_Z, attn_mask=mask_AZ)
        return H_A + H_A2L + H_A2Z

class ContextWeaver(nn.Module):
    """
    汇聚感知/地图/上下文为统一表征。
    输出: H_A, H_L, H_Z 及时序 tokens
    """
    def __init__(self, d: int = 192):
        super().__init__()
        self.agent_enc = AgentEncoder(d=d)
        self.lane_enc  = PolylineEncoder(d=d)
        self.zone_enc  = PolylineEncoder(d=d)
        self.fusion    = CrossFusion(d=d)
        self.ctx_fc    = nn.Linear(32, d)  # 天气/时段/地域等 context

    def forward(self, agent_seq, lane_polys, zone_polys, ctx_feat, masks=None):
        H_A, H_A_seq = self.agent_enc(agent_seq, mask=masks)
        H_L = self.lane_enc(lane_polys)
        H_Z = self.zone_enc(zone_polys)
        ctx = self.ctx_fc(ctx_feat).unsqueeze(1)  # [B,1,d]
        H_A = H_A + ctx
        H_L = H_L + ctx
        H_Z = H_Z + ctx
        H_A = self.fusion(H_A, H_L, H_Z, idx_AL=None, idx_AZ=None)  # 实际需邻域索引
        return {"H_A": H_A, "H_L": H_L, "H_Z": H_Z, "H_A_seq": H_A_seq}