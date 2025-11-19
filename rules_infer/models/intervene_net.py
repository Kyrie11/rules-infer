# -*- coding: utf-8 -*-
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentHead(nn.Module):
    """小闭集意图分类头（可选）。"""
    def __init__(self, d: int, n_intents: int):
        super().__init__()
        self.fc = nn.Linear(d, n_intents)

    def forward(self, h: torch.Tensor):
        return self.fc(h)  # logits

class Gate(nn.Module):
    """对候选原因 c 的门控 α_c ∈ [0,1]。"""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(8))  # 假设 8 个因素

    def forward(self):
        return torch.sigmoid(self.alpha)

def evidence_alignment(attn_map, evidence_mask):
    """证据对齐占位：计算注意/显著与 evidence 的覆盖率/IoU。"""
    # TODO: 实装覆盖率或 KL
    return torch.tensor(0.0, device=attn_map.device)

class InterveneNet(nn.Module):
    """
    输入: H_A/H_L/H_Z + z_i/z_g + 图消息 (由 SocioTopo-ALZ 提供)
    输出: 意图分布(或 CIG 参数) 与 反事实对比得分
    """
    def __init__(self, d: int, n_intents: int):
        super().__init__()
        self.intent_head = IntentHead(d, n_intents)
        self.gate = Gate()

    def forward(self, H: Dict, z_i: torch.Tensor, z_g: torch.Tensor, aux=None) -> Dict:
        hA = H["A"]  # [B, Na, d]
        cond = torch.cat([hA, z_i, z_g], dim=-1) if z_i is not None else hA
        logits = self.intent_head(cond)
        return {"logits_intent": logits}

    def counterfactual(self, inputs: Dict, interventions: List[Dict]) -> Dict:
        """
        inputs: 包含 H (node_feats), graph (structure), z_i, z_g
        """
        # 1. 基准前向
        with torch.no_grad():
            base_out = self.forward(inputs["H"], inputs["z_i"], inputs["z_g"])
            base_score = base_out["logits_intent"].softmax(-1).max(dim=-1).values # [B, Na]

        delta_scores = []

        for iv in interventions:
            # 2. 构建干预后的数据视图 (Do-Operator)
            # 注意：必须 Deep Copy 需要修改的部分，避免影响原始数据
            H_cf = {k: v.clone() for k, v in inputs["H"].items()}
            z_cf = inputs["z_i"].clone()

            if iv["type"] == "remove_agent":
                # 将指定 Agent 的特征置零，并使其在 Attention 中失效
                # 更好的做法是在 Graph 结构中删除节点，这里用 Masking 近似
                idx = iv["target_idx"]  # [B, idx]
                # 设为全0向量
                H_cf["A"][torch.arange(len(idx)), idx] = 0.0
                # 这是一个简化，实际上还需要在 Graph 的 Edge Mask 里把相关边去掉

            elif iv["type"] == "change_tl":
                # 修改 Zone 特征中的红绿灯状态位
                z_idx = iv["zone_idx"]
                new_state = iv["value"]  # e.g., one-hot vector
                H_cf["Z"][:, z_idx, -3:] = new_state

            # 3. 干预前向
            cf_out = self.forward(H_cf, z_cf, inputs["z_g"])
            cf_score = cf_out["logits_intent"].softmax(-1).max(dim=-1).values

            # 4. 计算因果效应 (Treatment Effect)
            effect = base_score - cf_score
            delta_scores.append(effect)

        return {"delta_scores": torch.stack(delta_scores, dim=1)}

def cf_directional_loss(score_base, score_cf, direction_label, margin=0.5):
    """
    方向性损失：若预期“去掉行人→让行概率下降”，则 score_base - score_cf 应 > margin。
    """
    # TODO: 按 label 聚合到 hinge/ranking
    return torch.tensor(0.0, device=score_base.device)
