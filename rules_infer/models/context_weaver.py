# batch_types.py
from typing import Optional, Dict, TypedDict, Tuple
import torch
from dataclasses import dataclass
import torch.nn as nn
from einops import rearrange



class Batch(TypedDict):
    # Agents
    agent_hist: torch.Tensor    # [B, Na, T, Fa], Fa≥ {x,y,yaw,vx,vy,ax,ay,omega, vis}
    agent_mask: torch.Tensor    # [B, Na, T]  1=valid, 0=pad
    agent_types: torch.Tensor   # [B, Na]     int enum (veh/ped/cyc/...)
    # Map - lanes (vectorized)
    lane_xy: torch.Tensor       # [B, Nl, Pmax, 2]
    lane_mask: torch.Tensor     # [B, Nl, Pmax]
    lane_attr: torch.Tensor     # [B, Nl, F_lattr]  # e.g., speed_limit, lane_type, priority
    # Map - zones (conflict/crosswalk/merge areas)
    zone_xy: torch.Tensor       # [B, Nz, Qmax, 2]
    zone_mask: torch.Tensor     # [B, Nz, Qmax]
    zone_types: torch.Tensor    # [B, Nz] int enum
    # Optional BEV
    bev: Optional[torch.Tensor] # [B, C, H, W] or None
    # Context (discrete ids)
    tod_id: torch.Tensor        # [B] time-of-day id
    weather_id: torch.Tensor    # [B]
    region_id: torch.Tensor     # [B]
    tl_state_id: torch.Tensor   # [B] (0: no_light/unknown etc.)
    # Neighborhood indices (precomputed by数据层, 半径裁剪+拓扑裁剪之后)
    a2l_idx: torch.Tensor       # [B, Na, K_l]  indices into Nl, -1=pad
    a2z_idx: torch.Tensor       # [B, Na, K_z]  indices into Nz, -1=pad
    a2a_idx: torch.Tensor       # [B, Na, K_a]  neighbor agents indices into Na, -1=pad
    # Pretrain targets (可选)
    lane_assoc_target: torch.Tensor  # [B, Na, T] 最近车道id或-1


@dataclass
class CWConfig:
    d_model: int = 192
    nhead: int = 8
    agent_encoder_layers: int = 3
    polyline_encoder_layers: int = 2
    cross_layers: int = 2
    ff_ratio: float = 4.0
    dropout: float = 0.1
    T: int = 8
    Pmax: int = 20
    Qmax: int = 20
    K_l: int = 8
    K_z: int = 4
    K_a: int = 12
    use_bev: bool = False
    bev_channels: int = 32      # 输出 token 维度
    num_tod: int = 6            # 清晨/早高峰/白天/晚高峰/夜间/未知
    num_weather: int = 6        # 晴/雨/雪/雾/干/未知
    num_region: int = 16
    num_tl: int = 5             # red/yellow/green/no_light/unknown
    conditioning_dim: int = 64  # 条件嵌入维度

class ContextConditioner(nn.Module):
    """
    输入离散 context ids -> 条件embedding -> 生成 FiLM 参数
    用于: (a) Agent/Map 编码时的 CondLayerNorm, (b) Cross-Attention 的bias
    """
    def __init__(self, num_tod, num_weather, num_region, num_tl, d_model, cond_dim):
        super().__init__()
        self.e_tod = nn.Embedding(num_tod, cond_dim)
        self.e_weather = nn.Embedding(num_weather, cond_dim)
        self.e_region = nn.Embedding(num_region, cond_dim)
        self.e_tl = nn.Embedding(num_tl, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim*4, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, d_model*2) # gamma,beta
        )

    def forward(self, tod_id, weather_id, region_id, tl_state_id):
        c = torch.cat([
            self.e_tod(tod_id), self.e_weather(weather_id),
            self.e_region(region_id), self.e_tl(tl_state_id)
        ], dim=-1)
        gb = self.mlp(c) # [B, 2*d]
        return gb.chunk(2, dim=-1) # gamma, beta  shaped [B, d_model]


class CondLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x, gamma, beta):
        # x: [..., d], gamma/beta: [B, d]  -> broadcast到batch维
        B = x.shape[0]
        y = self.ln(x)
        g = gamma.view(B, 1, -1)
        b = beta.view(B, 1, -1)
        return y * (1 + g) + b

class PosEnc1D(nn.Module):
    def __init__(self, d, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # [L, d]

    def forward(self, x):
        # x: [B, Na, T, d]
        T = x.size(2)
        return x + self.pe[:T].view(1,1,T,-1)

class AgentEncoder(nn.Module):
    def __init__(self, in_dim, d_model, nhead, nlayers, ff_ratio=4.0, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=int(d_model*ff_ratio), batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.pe = PosEnc1D(d_model)

    def forward(self, feats, mask=None, gamma=None, beta=None, cond_ln=None):
        """
        feats: [B, Na, T, Fa], mask: [B, Na, T]=1 valid
        """
        x = self.input(feats)             # [B, Na, T, d]
        x = self.pe(x)
        # 折叠 batch+agent 维度到 transformer
        B, Na, T, d = x.shape
        x = x.view(B*Na, T, d)
        key_padding = None
        if mask is not None:
            key_padding = (mask==0).view(B*Na, T)
        x = self.encoder(x, src_key_padding_mask=key_padding) # [B*Na, T, d]
        x = x.view(B, Na, T, d)
        if cond_ln is not None and gamma is not None:
            # 对时间维做条件归一（逐帧共享gamma/beta）
            x = cond_ln(x.view(B*Na, T, d), gamma.repeat_interleave(Na,0), beta.repeat_interleave(Na,0))
            x = x.view(B, Na, T, d)
        return x

class PointNet1D(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model//2), nn.ReLU(),
            nn.Linear(d_model//2, d_model), nn.ReLU()
        )
    def forward(self, x):
        # x: [B, N, P, Fin]
        return self.net(x)

class PolylineEncoder(nn.Module):
    """
    lane_xy: [B,Nl,P,2], lane_mask: [B,Nl,P]
    lane_attr: [B,Nl,Fl]
    -> lane tokens [B,Nl,d]
    同理 zone
    """
    def __init__(self, in_point_dim, in_attr_dim, d_model, nhead, nlayers, ff_ratio=4.0, dropout=0.1):
        super().__init__()
        self.penc = PointNet1D(in_point_dim, d_model)
        self.attr = nn.Linear(in_attr_dim, d_model) if in_attr_dim>0 else None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=int(d_model*ff_ratio), dropout=dropout
        )
        self.poly_enc = nn.TransformerEncoder(enc_layer, nlayers)

    def forward(self, xy, mask, attr=None):
        # 计算每段点特征（可添加方向/曲率/Δs）
        # 这里简化：仅 (x,y) -> MLP
        B, N, P, _ = xy.shape
        pts = self.penc(xy)                       # [B,N,P,d]
        key_pad = (mask==0).view(B*N, P)
        poly = self.poly_enc(pts.view(B*N,P,-1), src_key_padding_mask=key_pad)  # [B*N,P,d]
        # 池化到 polyline token（mask-aware mean）
        m = mask.view(B*N,P,1).float()
        token = (poly*m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        token = token.view(B, N, -1)
        if self.attr is not None and attr is not None:
            token = token + self.attr(attr)       # 简单融合属性
        return token  # [B,N,d]


class TopoEncoder(nn.Module):
    """
    输入：F_bev (或原始 bev 图像) -> 共享主干 -> lane/zone 两个解码头
    输出：H_L [B,Nl,d], H_Z [B,Nz,d], polyline/polygon 几何与掩码，实例数量可变
    """
    def __init__(self, d_model=192, n_queries_lane=64, n_queries_zone=32, token_dim=192):
        super().__init__()
        # 共享主干
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, d_model, 3, 2, 1), nn.ReLU()
        )  # 你也可以替换成 Swin/ConvNeXt/BEVFormer-feat

        # Lane queries + decoder
        self.lane_queries = nn.Parameter(torch.randn(n_queries_lane, d_model))
        self.lane_decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model * 4)
        self.lane_mask_head = nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.ReLU(),
                                            nn.Conv2d(d_model, n_queries_lane, 1))
        self.lane_attr_head = nn.Linear(d_model, 5)  #5代表lane类型数
        self.lane_geom_head = nn.Linear(d_model, 12)   # 折线/Bezier 参数（每段 3 个控制点）

        # Zone queries + decoder
        self.zone_queries = nn.Parameter(torch.randn(n_queries_zone, d_model))
        self.zone_decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model * 4)
        self.zone_mask_head = nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.ReLU(),
                                            nn.Conv2d(d_model, n_queries_zone, 1))
        self.zone_type_head = nn.Linear(d_model, 5)#5是zone类型
        self.zone_geom_head = nn.Linear(d_model, 16)   # polygon 参数（每个 zone 有多个点）

        # token 化：掩码池化 (mask-aware)
        self.token_proj = nn.Linear(d_model, token_dim)

    def _tokenize(self, feat_bev, attn_out, mask_logits):
        """
        feat_bev: [B,D,H,W]
        attn_out: [B,Q,D]  (query 的最终特征)
        mask_logits: [B,Q,H,W]
        输出每个实例的 token：对 feat_bev 做 mask-aware pooling
        """
        B, D, H, W = feat_bev.shape
        Q = attn_out.shape[1]
        masks = mask_logits.sigmoid()  # [B,Q,H,W]
        feat = feat_bev.unsqueeze(1)  # [B,1,D,H,W]
        masks = masks.unsqueeze(2)  # [B,Q,1,H,W]
        pooled = (feat * masks).sum(dim=(-1, -2)) / (masks.sum(dim=(-1, -2)).clamp_min(1e-5))  # [B,Q,D]
        token = self.token_proj(pooled) + self.token_proj(attn_out)  # 结合 query / pooled
        return token  # [B,Q,token_dim]

    def forward(self, bev_img):
        F = self.backbone(bev_img)  # [B,D,H',W']

        # Lane decode（cross-attn 到 F 的空间展开特征）
        lane_q = self.lane_queries.unsqueeze(0).expand(F.size(0), -1, -1)     # [B,Ql,D]
        lane_dec = self.lane_decoder(lane_q, F)                                # [B,Ql,D]
        lane_mask_logits = self.lane_mask_head(F)                              # [B,Ql,H',W']
        lane_token = self._tokenize(F, lane_dec, lane_mask_logits)             # [B,Ql,d]
        lane_attr = self.lane_attr_head(lane_dec)                              # [B,Ql,C_lane]
        lane_geom = self.lane_geom_head(lane_dec)                              # [B,Ql,G_lane]

        # Zone decode
        zone_q = self.zone_queries.unsqueeze(0).expand(F.size(0), -1, -1)
        zone_dec = self.zone_decoder(zone_q, F)
        zone_mask_logits = self.zone_mask_head(F)                              # [B,Qz,H',W']
        zone_token = self._tokenize(F, zone_dec, zone_mask_logits)             # [B,Qz,d]
        zone_type = self.zone_type_head(zone_dec)                              # [B,Qz,C_zone]
        zone_geom = self.zone_geom_head(zone_dec)                              # [B,Qz,G_zone]

        # 训练时用 Hungarian matching 将 (lane_mask, lane_geom, lane_type) 与 GT 对齐
        # 推理时按得分/面积/拓扑过滤得到 Nl/Nz 实例，再返回到 ContextWeaver
        return {
            "H_L": lane_token, "lane_mask": lane_mask_logits, "lane_attr_logits": lane_attr, "lane_geom": lane_geom,
            "H_Z": zone_token, "zone_mask": zone_mask_logits, "zone_type_logits": zone_type, "zone_geom": zone_geom
        }

class BEVEncoder(nn.Module):
    def __init__(self, in_ch, d_model):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, d_model, 3, 2, 1), nn.BatchNorm2d(d_model), nn.ReLU()
        )
        # 输出 tokens: [B, H', W', d] -> [B, Np, d]
    def forward(self, bev):
        f = self.backbone(bev)           # [B,d,H',W']
        B, d, Hp, Wp = f.shape
        tokens = f.permute(0,2,3,1).reshape(B, Hp*Wp, d)
        return tokens  # [B,Np,d]


class CrossBlock(nn.Module):
    """
    Q: [B, Nq, T?, d]  (agent 时间序列或池化后的agent)
    K/V: [B, Nk, d] or [B, Nk, d] (lane/zone tokens or bev tokens)
    这里我们以 per-agent 时间序列做 cross 到邻域 lanes/zones
    """
    def __init__(self, d_model, nhead, ff_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model*ff_ratio)), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(int(d_model*ff_ratio), d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, key_padding_mask=None):
        # Q: [B*Na, T, d], K/V: [B*Na, K, d]
        y, attn = self.attn(Q, K, V, key_padding_mask=key_padding_mask)  # attn: [B*Na, T, K]
        Q = self.ln1(Q + y)
        Q = self.ln2(Q + self.ff(Q))
        return Q, attn


# class ContextWeaver(nn.Module):
#     def __init__(self, cfg: CWConfig, fa: int, fl_attr: int, zone_point_feat: int=2):
#         """
#         fa: agent特征维度 (x,y,yaw,vx,vy,ax,ay,omega,vis,...)
#         fl_attr: lane属性维度（若无，传0）
#         zone_point_feat: zone点特征维度，默认仅(x,y)
#         """
#         super().__init__()
#         self.cfg = cfg
#         d = cfg.d_model
#         # 条件化
#         self.cond = ContextConditioner(cfg.num_tod, cfg.num_weather, cfg.num_region, cfg.num_tl, d, cfg.conditioning_dim)
#         self.cond_ln = CondLayerNorm(d)
#         # 编码器
#         self.agent_enc = AgentEncoder(fa, d, cfg.nhead, cfg.agent_encoder_layers, cfg.ff_ratio, cfg.dropout)
#         self.lane_enc = PolylineEncoder(in_point_dim=2, in_attr_dim=fl_attr, d_model=d,
#                                         nhead=cfg.nhead, nlayers=cfg.polyline_encoder_layers,
#                                         ff_ratio=cfg.ff_ratio, dropout=cfg.dropout)
#         self.zone_enc = PolylineEncoder(in_point_dim=zone_point_feat, in_attr_dim=0, d_model=d,
#                                         nhead=cfg.nhead, nlayers=cfg.polyline_encoder_layers,
#                                         ff_ratio=cfg.ff_ratio, dropout=cfg.dropout)
#         if cfg.use_bev:
#             self.bev_enc = BEVEncoder(in_ch=3, d_model=d)  # 视你的BEV通道
#         # Cross blocks
#         self.axl = nn.ModuleList([CrossBlock(d, cfg.nhead, cfg.ff_ratio, cfg.dropout) for _ in range(cfg.cross_layers)])
#         self.axz = nn.ModuleList([CrossBlock(d, cfg.nhead, cfg.ff_ratio, cfg.dropout) for _ in range(cfg.cross_layers)])
#
#     def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         B = batch['agent_hist'].shape[0]
#         # 条件
#         gamma, beta = self.cond(batch['tod_id'], batch['weather_id'], batch['region_id'], batch['tl_state_id'])
#         # 编码
#         # 编码 agent 特征（来自 BEV 特征或图像特征）
#         # 如果启用了 BEV，将图像特征经过 BEV 编码器，得到 BEV token
#         if self.cfg.use_bev and batch.get('bev', None) is not None:
#             bev_tokens = self.bev_enc(batch['bev'])  # [B, Np, d]
#         else:
#             bev_tokens = None
#         # Agent 编码：从 BEV tokens 中提取 agent 特征（如果使用 BEV）
#         H_A = self.agent_enc(bev_tokens, batch['agent_mask'], gamma, beta, self.cond_ln)   # [B,Na,T,d]
#         H_L = self.lane_enc(batch['lane_xy'], batch['lane_mask'], batch.get('lane_attr', None))     # [B,Nl,d]
#         H_Z = self.zone_enc(batch['zone_xy'], batch['zone_mask'], None)                             # [B,Nz,d]
#         # 邻域 gather：对每个 agent 取 K_l 条近邻车道，K_z 个近邻Zone
#         # 变形为 [B*Na, K, d]
#         Na = H_A.shape[1]
#         H_Ln, lk_mask = gather_index_2d(H_L, batch['a2l_idx'])  # ([B*Na,K_l,d], [B*Na,K_l])
#         H_Zn, zk_mask = gather_index_2d(H_Z, batch['a2z_idx'])
#         # 将 Agent 时间序列 [B,Na,T,d] -> [B*Na,T,d]
#         Q = H_A.reshape(B*Na, H_A.shape[2], H_A.shape[3])
#         # Cross A-L
#         attn_logs = {}
#         for i, blk in enumerate(self.axl):
#             Q, attn = blk(Q, H_Ln, H_Ln, key_padding_mask=(lk_mask==0))
#             attn_logs[f'axl_attn_{i}'] = attn  # [B*Na,T,K_l]
#         # Cross A-Z
#         for i, blk in enumerate(self.axz):
#             Q, attn = blk(Q, H_Zn, H_Zn, key_padding_mask=(zk_mask==0))
#             attn_logs[f'axz_attn_{i}'] = attn
#         # 回写：更新后的 agent 表征
#         H_A_fused = Q.view(B, Na, -1, self.cfg.d_model)  # [B,Na,T,d]
#
#         # 简单地 broadcast 给每体：Many-to-one 池化 or 追加为全局上下文；这里作为全局上下文缓存，供后续模块使用
#         return {
#             'H_A': H_A_fused,     # [B,Na,T,d]
#             'H_L': H_L,           # [B,Nl,d]
#             'H_Z': H_Z,           # [B,Nz,d]
#             'context_gb': (gamma, beta),
#             'attn_logs': attn_logs,
#             'bev_tokens': bev_tokens
#         }

class ContextWeaver(nn.Module):
    def __init__(self, cfg: CWConfig, fa: int,
                 topo: Optional[TopoEncoder] = None,
                 use_map_indices: bool = True):
        """
        fa: agent特征维度 (x,y,yaw,vx,vy,ax,ay,omega,vis,...)
        topo: 传入已构建好的 LearnedTopoEncoder；若 None 则内部构建一个默认的
        use_map_indices: True 时优先使用 batch 中的 a2l_idx/a2z_idx（与 HD map 对齐）；
                         False 时完全依赖 topo 的掩码在运行时构造邻域。
        """
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # 条件化
        self.cond = ContextConditioner(cfg.num_tod, cfg.num_weather, cfg.num_region, cfg.num_tl,
                                       d, cfg.conditioning_dim)
        self.cond_ln = CondLayerNorm(d)

        # 编码器
        self.agent_enc = AgentEncoder(fa, d, cfg.nhead, cfg.agent_encoder_layers, cfg.ff_ratio, cfg.dropout)

        if cfg.use_bev:
            self.bev_enc = BEVEncoder(in_ch=3, d_model=d)
        # LearnedTopoEncoder 负责 H_L / H_Z（从 BEV 学会划分）
        n_q_lane = getattr(cfg, "n_queries_lane", 64)
        n_q_zone = getattr(cfg, "n_queries_zone", 32)
        self.topo = TopoEncoder(d_model=d, n_queries_lane=n_q_lane,
                                n_queries_zone=n_q_zone, token_dim=d)

        # Cross blocks
        self.axl = nn.ModuleList([CrossBlock(d, cfg.nhead, cfg.ff_ratio, cfg.dropout)
                                  for _ in range(cfg.cross_layers)])
        self.axz = nn.ModuleList([CrossBlock(d, cfg.nhead, cfg.ff_ratio, cfg.dropout)
                                  for _ in range(cfg.cross_layers)])

        self.use_map_indices = use_map_indices

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch['agent_hist'].shape[0]
        gamma, beta = self.cond(batch['tod_id'], batch['weather_id'], batch['region_id'], batch['tl_state_id'])
        # ===== BEV 分支 =====
        if not self.cfg.use_bev or batch.get('bev', None) is None:
            raise ValueError("LearnedTopoEncoder 需要 BEV 图像输入。请设置 cfg.use_bev=True 并在 batch 中提供 'bev'。")

        # 1) Agent 的时序编码（保持你现有逻辑：从 BEV tokens/或其它来源抽取到 agent 时序特征）
        bev_tokens = self.bev_enc(batch['bev'])  # [B, Np, d] 作为 agent 编码输入的来源（与你现有实现一致）
        H_A = self.agent_enc(bev_tokens, batch['agent_mask'], gamma, beta, self.cond_ln)  # [B, Na, T, d]
        # 2) Lane / Zone 的学习划分与向量化 token（使用 LearnedTopoEncoder）
        topo_out = self.topo_enc(batch['bev'])
        H_L = topo_out["H_L"]  # [B, Ql, d]
        H_Z = topo_out["H_Z"]  # [B, Qz, d]
        lane_mask_logits = topo_out["lane_mask"]  # [B, Ql, H', W']
        zone_mask_logits = topo_out["zone_mask"]  # [B, Qz, H', W']
        lane_attr_logits = topo_out["lane_attr_logits"]  # [B, Ql, Cl]
        zone_type_logits = topo_out["zone_type_logits"]  # [B, Qz, Cz]
        lane_geom = topo_out["lane_geom"]  # [B, Ql, Gl]
        zone_geom = topo_out["zone_geom"]  # [B, Qz, Gz]

        # ===== 邻域 gather：对每个 agent 取 K_l 条近邻车道，K_z 个近邻 Zone =====
        Na = H_A.shape[1]
        H_Ln, lk_mask = gather_index_2d_safe(H_L, batch['a2l_idx'])  # ([B*Na,K_l,d], [B*Na,K_l])
        H_Zn, zk_mask = gather_index_2d_safe(H_Z, batch['a2z_idx'])  # ([B*Na,K_z,d], [B*Na,K_z])

        # 将 Agent 时间序列 [B,Na,T,d] -> [B*Na,T,d]
        Q = H_A.reshape(B * Na, H_A.shape[2], H_A.shape[3])

        # Cross A-L
        attn_logs = {}
        for i, blk in enumerate(self.axl):
            Q, attn = blk(Q, H_Ln, H_Ln, key_padding_mask=(lk_mask == 0))
            attn_logs[f'axl_attn_{i}'] = attn  # [B*Na, T, K_l]

        # Cross A-Z
        for i, blk in enumerate(self.axz):
            Q, attn = blk(Q, H_Zn, H_Zn, key_padding_mask=(zk_mask == 0))
            attn_logs[f'axz_attn_{i}'] = attn  # [B*Na, T, K_z]

        # 回写：更新后的 agent 表征
        H_A_fused = Q.view(B, Na, -1, self.cfg.d_model)  # [B,Na,T,d]

        # 组织输出：保留用于可视化/调试的 topo 辅助结果（掩码/几何/类型）
        topo_aux = {
            'lane_mask_logits': lane_mask_logits,
            'zone_mask_logits': zone_mask_logits,
            'lane_attr_logits': lane_attr_logits,
            'zone_type_logits': zone_type_logits,
            'lane_geom': lane_geom,
            'zone_geom': zone_geom
        }

        return {
            'H_A': H_A_fused,  # [B,Na,T,d]
            'H_L': H_L,  # [B,Ql,d]（LearnedTopoEncoder 产出）
            'H_Z': H_Z,  # [B,Qz,d]
            'context_gb': (gamma, beta),
            'attn_logs': attn_logs,
            'bev_tokens': bev_tokens,
            'topo_aux': topo_aux  # 用于可视化 attention/掩码/几何
        }

    def gather_index_2d_safe(tokens: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更稳健的 gather：同时检查上下界，避免预测实例数(Ql/Qz)与索引不一致时越界。
        tokens: [B, N, d]
        idx:    [B, M, K]  (例如 B×Na×K_l 的 lane/zone 索引；-1 表示 padding)
        返回:
          gathered:   [B*M, K, d]
          valid_mask: [B*M, K]  （True=有效）
        """
        B, N, d = tokens.shape
        BM, K = idx.shape[0] * idx.shape[1], idx.shape[2]
        flat_idx = idx.view(B * idx.shape[1], K)

        # 有效性：[-1 为无效] 以及 [小于 N]
        valid = (flat_idx >= 0) & (flat_idx < N)
        clamped = flat_idx.clamp(min=0, max=max(N - 1, 0))

        # 扩展批维
        b_ids = torch.arange(B, device=tokens.device).view(B, 1, 1).expand_as(idx[:, :, :1]).reshape(B * idx.shape[1],
                                                                                                     1)
        b_ids = b_ids.expand(BM, K)

        gathered = tokens[b_ids, clamped, :]  # [B*M, K, d]
        gathered = gathered * valid.unsqueeze(-1).float()
        return gathered, valid

def gather_index_2d(tokens: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    tokens: [B, N, d]
    idx:    [B, M, K]  (例如 B×Na×K_l 的 lane 索引)
    返回:
      gathered: [B*M, K, d]
      valid_mask: [B*M, K]
    """
    B, N, d = tokens.shape
    BM, K = idx.shape[0]*idx.shape[1], idx.shape[2]
    flat_idx = idx.view(B*idx.shape[1], K)
    valid = (flat_idx >= 0)
    clamped = flat_idx.clamp(min=0)
    # 扩展批维
    b_ids = torch.arange(B, device=tokens.device).view(B,1,1).expand_as(idx[:,:, :1]).reshape(B*idx.shape[1],1)
    b_ids = b_ids.expand(BM, K)
    gathered = tokens[b_ids, clamped, :]  # [B*M,K,d]
    # 对无效(-1)位置置零
    gathered = gathered * valid.unsqueeze(-1).float()
    return gathered, valid

def last_valid_xy(agent_hist: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    """
    agent_hist: [B, Na, T, Fa], 其中前两维为 (x,y) 或你在数据管道保证了对应下标
    agent_mask: [B, Na, T]
    返回每个 agent 在最后有效帧的 (x,y): [B, Na, 2]
    """
    B, Na, T, F = agent_hist.shape
    m = agent_mask.bool()                                    # [B,Na,T]
    # 找最后一个有效t索引
    idx = (m.long() * torch.arange(T, device=m.device)).argmax(dim=-1)  # [B,Na]
    xy = torch.gather(agent_hist[..., :2], dim=2, index=idx.unsqueeze(-1).unsqueeze(-1).expand(B, Na, 1, 2))
    return xy.squeeze(2)  # [B,Na,2]

def world_to_bev_xy(xy_world: torch.Tensor, bev_meta: Dict) -> torch.Tensor:
    """
    将世界坐标映射到 BEV 像素坐标（用于从 mask 上采样）。
    bev_meta 需提供:
      - 'origin': (x0, y0)  世界坐标对应 BEV(0,0) 的原点
      - 'resolution': meters_per_pixel (float) 或 (rx, ry)
      - 'size': (H', W')  对应 LearnedTopoEncoder backbone 输出的掩码尺寸
    返回整数像素坐标 [B,Na,2]，顺序为 (u=W-axis, v=H-axis)。
    """
    x0, y0 = bev_meta['origin']
    res = bev_meta['resolution']
    Hp, Wp = bev_meta['size']
    if isinstance(res, (tuple, list)):
        rx, ry = res
    else:
        rx = ry = float(res)

    x, y = xy_world[..., 0], xy_world[..., 1]
    # 将世界坐标平移并按分辨率缩放
    u = ((x - x0) / rx).round().long()
    v = ((y - y0) / ry).round().long()
    # 限制在图像边界
    u = u.clamp(min=0, max=Wp - 1)
    v = v.clamp(min=0, max=Hp - 1)
    return torch.stack([u, v], dim=-1)  # [B,Na,2]

def topk_queries_from_masks_at_points(mask_logits: torch.Tensor,
                                      pts_uv: torch.Tensor,
                                      K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据 agent 的 BEV 像素坐标，在每个 query 的 mask 上取该点处的响应，选 Top-K query 作为邻域。
    mask_logits: [B, Q, H', W']
    pts_uv:      [B, Na, 2]  (u,v) 像素坐标
    返回:
      idx:  [B, Na, K]  (query 索引)
      valid:[B, Na, K]  (这里都为 True；若需剔除低响应可再加阈值)
    """
    B, Q, Hp, Wp = mask_logits.shape
    B2, Na, _ = pts_uv.shape
    assert B == B2
    # 采样该点处每个 query 的 mask 值
    u = pts_uv[..., 0]  # [B,Na]
    v = pts_uv[..., 1]  # [B,Na]
    # 索引: 对每个 b,n，取 mask_logits[b, :, v, u] -> [Q]
    scores = []
    for b in range(B):
        sb = mask_logits[b, :, v[b], u[b]]  # [Q,Na]
        scores.append(sb.t())               # [Na,Q]
    scores = torch.stack(scores, dim=0)     # [B,Na,Q]
    topk = torch.topk(scores, k=min(K, Q), dim=-1)  # values, indices
    idx = topk.indices  # [B,Na,K]
    valid = torch.ones_like(idx, dtype=torch.bool)
    return idx, valid
