"""Decoupled span + objectness head.

1-D adaptation of YOLOX's decoupled detection head (Megvii YOLOX, Apache-2.0): YOLOX found that
giving classification and localization *separate* branches (rather than one shared head) improves
detection. Here, per typed query, one branch regresses the in-read span (start/end, as normalized
fractions via sigmoid — DETR box convention) and a separate branch predicts objectness (is this
gene present in the read). See sota/ATTRIBUTION.md.
"""
import torch
import torch.nn as nn


class SpanHead(nn.Module):
    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__()
        h = hidden or d_model
        # decoupled: independent stems for regression and objectness (YOLOX)
        self.reg_stem = nn.Sequential(nn.Linear(d_model, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.obj_stem = nn.Sequential(nn.Linear(d_model, h), nn.SiLU())
        self.reg = nn.Linear(h, 2)     # (start, end) as normalized fractions of the read length
        self.obj = nn.Linear(h, 1)     # objectness logit (presence)

    def forward(self, q: torch.Tensor) -> dict:
        """q (..., d) per-query rep -> {'span': (..., 2) in [0,1], 'objectness': (...) logit}."""
        span = torch.sigmoid(self.reg(self.reg_stem(q)))     # DETR-style normalized box coords
        obj = self.obj(self.obj_stem(q)).squeeze(-1)
        return {"span": span, "objectness": obj}
