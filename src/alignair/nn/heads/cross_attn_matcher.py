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

    @staticmethod
    def _first_last(mask):
        # mask (N,S) bool -> (first_idx, last_idx) (N,) of valid positions
        N, S = mask.shape
        ar = torch.arange(S, device=mask.device)
        first = torch.where(mask, ar, torch.full_like(ar, S)).min(dim=1).values.clamp(max=S - 1)
        last = torch.where(mask, ar, torch.full_like(ar, -1)).max(dim=1).values.clamp(min=0)
        return first, last

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
        # germline coords: the first/last valid seg token's attention distribution over germline
        # (its 5' end maps to germline_start, its 3' end to germline_end). argmax over Lg = coord.
        logw = torch.log(wmean.clamp_min(1e-9))                         # (BC,S,Lg) log-attention
        logw = logw.masked_fill(~gm.unsqueeze(1), -1e9)                # respect germline mask
        first, last = self._first_last(sm)                             # (BC,), (BC,)
        bc = torch.arange(B * C, device=seg_reps.device)
        gstart = logw[bc, first].reshape(B, C, Lg)
        gend = logw[bc, last].reshape(B, C, Lg)
        return match, gstart, gend


def xattn_match(matcher, seg_reps, seg_mask, pos_reps, pos_mask, cand_idx):
    """Gather a per-read candidate pool's germline reps, run the CrossAttnMatcher, and decode
    allele logits + germline coords + the chosen global allele index.

    seg_reps (B,S,d), seg_mask (B,S); pos_reps (K,Lg,d), pos_mask (K,Lg); cand_idx (B,C) long.
    Returns dict(allele_logits, best_idx, best_global_idx, germ_start, germ_end,
                 gstart_logits, gend_logits)."""
    B, C = cand_idx.shape
    cand_reps = pos_reps[cand_idx]                                # (B,C,Lg,d)
    cand_mask = pos_mask[cand_idx]                                # (B,C,Lg)
    match, gstart, gend = matcher(seg_reps, seg_mask, cand_reps, cand_mask)
    bi = torch.arange(B, device=cand_idx.device)
    best = match.argmax(dim=1)                                    # (B,)
    return {
        "allele_logits": match,
        "best_idx": best,
        "best_global_idx": cand_idx[bi, best],
        "germ_start": gstart[bi, best].argmax(dim=-1),
        "germ_end": gend[bi, best].argmax(dim=-1),
        "gstart_logits": gstart,
        "gend_logits": gend,
        "pool_idx": cand_idx,
    }
