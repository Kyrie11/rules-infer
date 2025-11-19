# -*- coding: utf-8 -*-
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_ops import HeteroGraph, gumbel_softmax_sample

class EdgeTypePredictor(nn.Module):
    """
    对潜在关系边 (A-A, A-Z) 预测类型分布 π_e。
    """
    def __init__(self, d_e: int, n_types: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_e, 128), nn.ReLU(), nn.Linear(128, n_types))

    def forward(self, e_feat: torch.Tensor, tau: float=1.0, hard: bool=False):
        logits = self.mlp(e_feat)
        pi = gumbel_softmax_sample(logits, tau=tau, hard=hard)  # [E, K]
        return pi, logits

def antisymmetry_penalty(pi_yield: torch.Tensor, idx_src: torch.Tensor, idx_dst: torch.Tensor):
    """
    yield_to 反对称: p(u->v) 与 p(v->u) 不应同时大。
    伪代码：对互逆边做 KL 或 L1 惩罚。
    """
    # TODO: 构造互逆配对，计算 penalty
    return torch.tensor(0.0, device=pi_yield.device)

def transitivity_penalty_follow(pi_follow: torch.Tensor, triplets):
    """
    follow 传递性弱约束: u->v 且 v->w 则 u->w 倾向成立。
    """
    # TODO: 对三元组构造 soft 约束
    return torch.tensor(0.0, device=pi_follow.device)

class RelAttention(nn.Module):
    """关系特定注意(可加入风险/优先级偏置)。"""
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.scale = d ** -0.5
        # 可学习偏置系数 β
        self.beta_risk = nn.Parameter(torch.tensor(0.1))
        self.beta_prio = nn.Parameter(torch.tensor(0.1))
        self.beta_impl = nn.Parameter(torch.tensor(0.1))

    def forward(self, q_in, k_in, v_in, phi_risk=None, phi_prio=None, phi_impl=None):
        Q = self.q(q_in); K = self.k(k_in); V = self.v(v_in)
        att = (Q @ K.transpose(-2, -1)) * self.scale
        if phi_risk is not None: att = att + self.beta_risk * phi_risk
        if phi_prio is not None: att = att + self.beta_prio * phi_prio
        if phi_impl is not None: att = att - self.beta_impl * phi_impl
        w = att.softmax(-1)
        return w @ V

class HGTLayer(nn.Module):
    """简化 HGT：针对 (A-A/A-L/A-Z/L-Z) 分别建投影与注意。"""
    def __init__(self, d: int):
        super().__init__()
        self.rel_attn = nn.ModuleDict({
            "A-A": RelAttention(d),
            "A-L": RelAttention(d),
            "A-Z": RelAttention(d),
            "L-Z": RelAttention(d),
        })
        self.ffn = nn.ModuleDict({
            "A": nn.Sequential(nn.Linear(d, 2*d), nn.ReLU(), nn.Linear(2*d, d)),
            "L": nn.Sequential(nn.Linear(d, 2*d), nn.ReLU(), nn.Linear(2*d, d)),
            "Z": nn.Sequential(nn.Linear(d, 2*d), nn.ReLU(), nn.Linear(2*d, d)),
        })

        self.risk_encoder = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))  # 输入 [TTC, TTZ, Dist]

    def forward(self, G: HeteroGraph) -> Dict:
        H = G.node_feats  # dict
        # 伪实现：对每种边关系做一次全连接注意并残差写回
        for rel, attn in self.rel_attn.items():
            if rel not in G.edges: continue
            src_type, dst_type = rel.split("-")
            # 1. 获取索引和特征
            edge_idx = G.edges[rel]  # [2, E]
            src_idxs, dst_idxs = edge_idx[0], edge_idx[1]

            h_src = H[src_type]  # [B, N_src, d] -> 需 gather 到边
            h_dst = H[dst_type]  # [B, N_dst, d]
            # Gather 逻辑 (简化版，假设 Batch=1 或已处理 Batch Offset)
            # 实际实现推荐使用 torch_geometric 的 MessagePassing 机制
            # 这里用 tensor 操作演示逻辑：
            q_in = h_dst[dst_idxs]  # [E, d]
            k_in = h_src[src_idxs]  # [E, d]
            v_in = h_src[src_idxs]  # [E, d]
            # 2. 处理风险偏置 (Risk Bias)
            phi_risk = None
            if rel in G.edge_attr:
                # 假设 edge_attr[rel] 包含 [TTC, TTZ, Priority]
                attrs = G.edge_attr[rel]
                # 提取 TTC/TTZ 相关列进行编码
                risk_feat = attrs[:, :3]
                phi_risk = self.risk_encoder(risk_feat)  # [E, 1]
            # 3. Attention 计算 (Point-wise for edges, then scatter sum)
            # 注意：这里不能直接用 MultiheadAttention，因为是 Sparse Graph
            # 我们需要手写简易的 Sparse Attention 或使用 torch_geometric.nn.TransformerConv
            # --- 简易 Sparse Attention 实现 ---
            msg = attn.sparse_forward(q_in, k_in, v_in, edge_idx, phi_risk=phi_risk)

        for t in H.keys():
            H[t] = H[t] + self.ffn[t](H[t])
        return H


class SocioTopoALZ(nn.Module):
    """整合：潜在边类型学习 + HGT 消息传递 + Zone 超边聚合（留接口）"""
    def __init__(self, d: int, edge_types_AA: int=6, edge_types_AZ: int=5):
        super().__init__()
        self.edge_pred_AA = EdgeTypePredictor(d_e=d*2+16, n_types=edge_types_AA)
        self.edge_pred_AZ = EdgeTypePredictor(d_e=d*2+16, n_types=edge_types_AZ)
        self.hgt = HGTLayer(d=d)

    def forward(self, G: HeteroGraph, tau: float=1.0) -> Dict:
        # === 1) 潜在边类型 ===
        # 伪代码：拼边特征 e_feat = [h_src, h_dst, geo/risk attrs]
        # pi_AA, logits_AA = self.edge_pred_AA(e_feat_AA, tau, hard=False)
        # pi_AZ, logits_AZ = self.edge_pred_AZ(e_feat_AZ, tau, hard=False)
        pi_AA, logits_AA = None, None
        pi_AZ, logits_AZ = None, None

        # === 2) HGT 消息传递（带注意偏置：在 HGTLayer 内实现）===
        H = self.hgt(G)

        # === 3) 正则项（反对称/传递/稀疏度）===
        loss_prior = torch.tensor(0.0, device=H["A"].device)
        # TODO: antisymmetry_penalty / transitivity_penalty / L1-sparsity

        return {"H": H, "pi_AA": pi_AA, "pi_AZ": pi_AZ,
                "logits_AA": logits_AA, "logits_AZ": logits_AZ,
                "loss_prior": loss_prior}
