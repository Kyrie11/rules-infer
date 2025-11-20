# modules/heads_pretrain.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedTrajReconstructHead(nn.Module):
    """
    在 agent_hist 上随机mask若干帧的位置/速度，要求从 ContextWeaver 融合后的 H_A 复原
    """
    def __init__(self, d_model, out_dim=4):  # 重建 [x,y,vx,vy] 可扩展
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, out_dim)
        )
    def forward(self, H_A, target, mask_positions):
        # H_A: [B,Na,T,d], target: [B,Na,T,out_dim], mask_positions: [B,Na,T] 1=需要监督
        pred = self.proj(H_A)
        loss = F.smooth_l1_loss(pred[mask_positions>0], target[mask_positions>0])
        return loss, pred

class LaneAssocHead(nn.Module):
    """
    车道关联分类：给每个 agent 每帧预测当前关联lane id（或 NONE）
    """
    def __init__(self, d_model, num_lanes, none_idx=-1):
        super().__init__()
        self.num_lanes = num_lanes
        self.none_idx = none_idx
        self.cls = nn.Linear(d_model, num_lanes+1)  # +1 作为 NONE
    def forward(self, H_A, target):
        # target: [B,Na,T] in [0..num_lanes-1] or -1
        B,Na,T,_ = H_A.shape
        logits = self.cls(H_A)     # [B,Na,T,num_lanes+1]
        target_ = target.clone()
        target_[target_<0] = self.num_lanes  # 将 -1 映射到 NONE 类
        loss = nn.CrossEntropyLoss(ignore_index=self.num_lanes)(logits.view(-1, self.num_lanes+1),
                                                                target_.view(-1))
        return loss, logits

if __name__=="__main__":
    cfg = CWConfig()
    model = ContextWeaver(cfg, fa=9, fl_attr=4)  # 例如 agent feat 9维，lane attr 4维

    out = model(batch)     # 见 Batch 契约
    H_A, H_L, H_Z = out['H_A'], out['H_L'], out['H_Z']

    # 预训练
    mask_head = MaskedTrajReconstructHead(cfg.d_model, out_dim=4)
    lane_head = LaneAssocHead(cfg.d_model, num_lanes=batch['lane_xy'].shape[1])

    L_mask, _ = mask_head(H_A, target=batch['agent_hist'][..., :4], mask_positions=batch['agent_mask'])
    L_assoc, _ = lane_head(H_A, target=batch['lane_assoc_target'])
    loss = L_mask + 0.5*L_assoc
    loss.backward()