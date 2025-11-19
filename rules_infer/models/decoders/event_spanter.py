# -*- coding: utf-8 -*-
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class EventSpanter(nn.Module):
    """
    事件分类 + span 边界回归 + 角色识别（Hungarian 指派可在外部 utils 实现）
    """
    def __init__(self, d: int, n_events: int, n_roles: int):
        super().__init__()
        self.event_head = nn.Linear(d, n_events)
        self.span_head  = nn.Linear(d, 2)       # start/end 相对指针
        self.role_head  = nn.Linear(d, n_roles) # 多标签

    def forward(self, H_A_seq) -> Dict:
        # H_A_seq: [B, Na, T, d]  —— 可按片段聚合或直接时序预测
        B, Na, T, d = H_A_seq.shape
        x = H_A_seq.mean(2)         # 简化：时间平均；实际可用 Bi-Transformer/指针网络
        event_logits = self.event_head(x)
        role_logits  = self.role_head(x)
        span = torch.sigmoid(self.span_head(x)) # 相对 0..1
        return {"event_logits": event_logits, "role_logits": role_logits, "span": span}
