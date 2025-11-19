# -*- coding: utf-8 -*-
from typing import Dict
import torch
import torch.nn as nn
from utils.soft_logic import satisfaction_yield_to, rule_penalty, soft_and

class PointerNet(nn.Module):
    """从候选集合中指向 target（简化骨架）。"""
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)

    def forward(self, q_repr, cand_repr):
        score = (self.q(q_repr) * self.k(cand_repr)).sum(-1)  # [B, Na, Nc]
        prob  = score.softmax(-1)
        idx   = prob.argmax(-1)
        return idx, prob

class CIGComposer(nn.Module):
    """
    输出若干三元组 (relation, target, goal_params) 的参数。
    relation: 分类头; target: 指针; goal: 连续参数(如期望间隙/速度).
    """
    def __init__(self, d: int, n_rel: int):
        super().__init__()
        self.rel_head = nn.Linear(d, n_rel)
        self.ptr      = PointerNet(d)
        self.goal_mlp = nn.Linear(d, 3)  # 例: [desired_gap, desired_speed, stop_dist]

    def forward(self, H: Dict) -> Dict:
        hA = H["A"]  # [B, Na, d]
        rel_logits = self.rel_head(hA)
        # target：以邻居 agent/Zone/Lane 表征为候选；此处骨架用自身表征代替
        idx, prob = self.ptr(hA, hA)
        goal = self.goal_mlp(hA)
        return {"rel_logits": rel_logits, "target_idx": idx, "target_prob": prob, "goal": goal}

    def rule_satisfaction(self, trajs, neighbors, cig_pred) -> torch.Tensor:
        """
        trajs: Ego 预测轨迹 [B, Na, T, 2] (取 M 条中概率最大的一条或加权)
        target_trajs: [B, Na, T, 2] (目标 Agent 的轨迹，来自 GT 或预测)
        cig_pred: {"rel_logits": ..., "target_idx": ...}
        """
        B, Na, _ = cig_pred["rel_logits"].shape
        pred_rel = cig_pred["rel_logits"].argmax(-1) # [B, Na]
        target_idx = cig_pred["target_idx"] # [B, Na]

        penalties = []

        for b in range(B):
            for i in range(Na):
                rel = pred_rel[b, i]
                if rel == 0: # 假设 0 是 "Yield"
                    tgt_id = target_idx[b, i]
                    # 提取轨迹
                    ego_tr = trajs[b, i]
                    tgt_tr = target_trajs[b, tgt_id] # 需注意 target_trajs 的索引映射

                    # 计算满足度
                    sat = satisfaction_yield_to(ego_tr, tgt_tr, tau=2.0)

                    # 我们希望 sat 接近 1，所以 penalty = 1 - sat
                    penalties.append(1.0 - sat)

        if not penalties:
            return torch.tensor(0.0, device=trajs.device)
        return torch.stack(penalties).mean()