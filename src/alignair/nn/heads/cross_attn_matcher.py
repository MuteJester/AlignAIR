from __future__ import annotations
import torch
import torch.nn as nn


class CrossAttnMatcher(nn.Module):
    """Token-level read-segment x candidate-germline cross-attention. For each of C candidate
    germlines per read, the read-segment tokens (queries) attend to that germline's tokens (keys);
    a ColBERT-style MaxSim (each segment token's best germline similarity, masked-averaged) is the
    allele match score, and the boundary tokens' attention distribution gives germline start/end
    pointers. Learnable through the q/k projections; the MaxSim inductive bias ranks the true
    germline highest even before training."""

    def __init__(self, d_model: int, nhead: int = 8):
        super().__init__()
        assert d_model % nhead == 0
        self.h, self.hd = nhead, d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.scale = nn.Parameter(torch.ones(()))           # learnable match temperature

    def _attend(self, seg, sm, germ, gm):
        # seg (N,S,d), germ (N,Lg,d) -> wmean (N,S,Lg) softmax attn, simmax (N,S) MaxSim per seg pos
        N, S, d = seg.shape
        Lg = germ.shape[1]
        Q = self.q(seg).view(N, S, self.h, self.hd).transpose(1, 2)     # (N,h,S,hd)
        K = self.k(germ).view(N, Lg, self.h, self.hd).transpose(1, 2)
        att = (Q @ K.transpose(-1, -2)) / (self.hd ** 0.5)             # (N,h,S,Lg)
        att = att.masked_fill(~gm[:, None, None, :], -1e9)
        w = torch.softmax(att, dim=-1).mean(1)                          # (N,S,Lg) head-averaged attn
        simmax = att.amax(dim=-1).mean(1)                              # (N,S) best germline match/pos
        return w, simmax

    def forward(self, seg_reps, seg_mask, cand_reps, cand_mask):
        B, S, d = seg_reps.shape
        C, Lg = cand_reps.shape[1], cand_reps.shape[2]
        seg = seg_reps.unsqueeze(1).expand(B, C, S, d).reshape(B * C, S, d)
        sm = seg_mask.unsqueeze(1).expand(B, C, S).reshape(B * C, S)
        germ = cand_reps.reshape(B * C, Lg, d)
        gm = cand_mask.reshape(B * C, Lg)
        wmean, simmax = self._attend(seg, sm, germ, gm)                 # (BC,S,Lg), (BC,S)
        simmax = simmax.masked_fill(~sm, 0.0)
        match = (self.scale * simmax.sum(-1) / sm.sum(-1).clamp(min=1)).reshape(B, C)
        gstart = torch.zeros(B, C, Lg, device=seg_reps.device)         # filled in Task 2
        gend = torch.zeros(B, C, Lg, device=seg_reps.device)
        return match, gstart, gend
