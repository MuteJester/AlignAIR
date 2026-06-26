"""Raw-nucleotide base-match utilities shared across germline aligners.

The base-match channel is the dynamic-genotype floor: a +1 match / -1 mismatch signal on
raw ACGT tokens that is always correct even for a germline never seen in training, so a
novel/dynamic allele still aligns on real bases.
"""
import math

import torch
import torch.nn.functional as F


def base_match_matrix(seg_tok: torch.Tensor, germ_tok: torch.Tensor) -> torch.Tensor:
    """Raw +1 match / -1 mismatch / 0 non-ACGT base-match grid (B,S,Lg)."""
    st, gt = seg_tok.unsqueeze(2), germ_tok.unsqueeze(1)            # (B,S,1),(B,1,Lg)
    real = (st >= 1) & (st <= 4) & (gt >= 1) & (gt <= 4)
    return real.float() * (2.0 * (st == gt).float() - 1.0)


def base_match_channel(M: torch.Tensor, seg_tok, germ_tok, seg_reliability,
                       match_weight: torch.Tensor, match_floor: float) -> torch.Tensor:
    """Add the SNP base-match term to score matrix M (B,S,Lg). Pass-through when tokens are
    absent. seg_tok (B,S), germ_tok (B,Lg), seg_reliability (B,S) in [0,1] down-weights the
    channel at likely-SHM positions (so a substitution does not penalise the true allele)."""
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
