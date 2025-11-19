# -*- coding: utf-8 -*-
from typing import Dict
import torch

class OmniLoss:
    def __init__(self, w=None):
        self.w = w or dict(traj=1.0, event=1.0, edge=0.5, cf=0.5, style=0.5, plan=0.5)

    def __call__(self, parts: Dict) -> Dict:
        """
        parts: dict 收集各子模块损失 (可能为 None)：
          - traj_loss, event_loss, edge_prior, cf_loss, style_loss, plan_loss
          - 以及各自的指标/分项
        """
        loss = 0.0
        if parts.get("traj_loss") is not None:  loss += self.w["traj"]  * parts["traj_loss"]
        if parts.get("event_loss") is not None: loss += self.w["event"] * parts["event_loss"]
        if parts.get("edge_prior") is not None: loss += self.w["edge"]  * parts["edge_prior"]
        if parts.get("cf_loss") is not None:    loss += self.w["cf"]    * parts["cf_loss"]
        if parts.get("style_loss") is not None: loss += self.w["style"] * parts["style_loss"]
        if parts.get("plan_loss") is not None:  loss += self.w["plan"]  * parts["plan_loss"]
        return {"loss": loss}
