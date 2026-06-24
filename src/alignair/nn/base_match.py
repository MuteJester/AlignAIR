"""Shared SNP-sensitive base-match channel for germline aligners.

Extracted verbatim from SoftDPAligner._scores so the soft-DP and the pointer
aligner use identical emission math: a raw-ACGT +1/-1 match term (floored so a
never-seen germline still aligns on real bases — the dynamic-genotype guarantee),
optionally state-conditioned by a per-read-position reliability that down-weights
likely-SHM positions (so a substitution does not penalise the true allele).
"""
import math

import torch
import torch.nn.functional as F


def base_match_channel(M: torch.Tensor, seg_tok, germ_tok, seg_reliability,
                       match_weight: torch.Tensor, match_floor: float) -> torch.Tensor:
    """Add the SNP base-match term to score matrix M (B,S,Lg). Pass-through when
    tokens are absent. seg_tok (B,S), germ_tok (B,Lg), seg_reliability (B,S) in [0,1]."""
    if seg_tok is None or germ_tok is None:
        return M
    st, gt = seg_tok.unsqueeze(2), germ_tok.unsqueeze(1)            # (B,S,1),(B,1,Lg)
    real = (st >= 1) & (st <= 4) & (gt >= 1) & (gt <= 4)           # ACGT only
    u = real.float() * (2.0 * (st == gt).float() - 1.0)           # +1 match / -1 mismatch
    lam = match_floor + F.softplus(match_weight)                   # scalar coefficient
    if seg_reliability is not None:
        a = (lam * seg_reliability.clamp(0.0, 1.0)).unsqueeze(2)   # (B,S,1)
        norm = torch.log(a.exp() + 3.0 * (-a).exp()) - math.log(4.0)  # (B,S,1)
        return M + a * u - norm * real.float()
    return M + lam * u
