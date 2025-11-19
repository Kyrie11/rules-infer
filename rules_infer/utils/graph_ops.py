# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

def gumbel_softmax_sample(logits: torch.Tensor, tau: float=1.0, hard: bool=False):
    g = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
    y = F.softmax((logits + g)/tau, dim=-1)
    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y

class HeteroGraph:
    """
    简化版异质图容器：节点特征 dict，边索引 dict，边特征 dict。
    node_feats: {"A": [B, Na, d], "L": [B, Nl, d], "Z":[B, Nz, d]}
    edges: {("A","A"): (idx_src, idx_dst), ("A","Z"):...}  # 每批次内部索引
    """
    def __init__(self, node_feats: Dict, edges: Dict, edge_attr: Dict):
        self.node_feats = node_feats
        self.edges = edges
        self.edge_attr = edge_attr
