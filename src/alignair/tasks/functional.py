"""Shared numerical primitives for boundary and weighted task losses."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ..neural.weighting import UncertaintyWeight


def soft_gaussian_target(gt: torch.Tensor, length: int, sigma: float = 1.5) -> torch.Tensor:
    """Discretized Gaussian over positions ``0..length-1`` centered at ``gt``, renormalized."""
    gt = gt.reshape(-1, 1).clamp(0, length - 1)
    idx = torch.arange(length, device=gt.device, dtype=torch.float32).unsqueeze(0)   # (1, L)
    return F.softmax(-0.5 * (idx - gt) ** 2 / sigma ** 2, dim=-1)                     # (B, L)


def _kendall(uw: UncertaintyWeight, loss: torch.Tensor, row_mask=None) -> torch.Tensor:
    weighted = loss * uw() + uw.penalty()
    return weighted if row_mask is None else weighted * row_mask.any().to(weighted.dtype)


def _mean_rows(values, row_mask=None):
    rows = values.reshape(values.shape[0], -1).mean(-1)
    if row_mask is None:
        return rows.mean()
    weights = row_mask.reshape(-1).to(rows.dtype)
    return (rows * weights).sum() / weights.sum().clamp_min(1)


def soft_cross_entropy_rows(logits: torch.Tensor, soft_target: torch.Tensor,
                            mask: torch.Tensor | None = None) -> torch.Tensor:
    """Soft-label cross-entropy over positions. ``mask`` (B, L bool) restricts the softmax to the
    valid read positions so the model never competes on padding; the pad terms are dropped from the
    sum (guarding against ``0 * -inf`` NaN)."""
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    logp = F.log_softmax(logits, dim=-1)
    term = soft_target * logp
    if mask is not None:
        term = torch.where(mask, term, term.new_zeros(()))
    return -term.sum(-1)


def _soft_ce(logits: torch.Tensor, soft_target: torch.Tensor,
             mask: torch.Tensor | None = None, row_mask=None) -> torch.Tensor:
    """Mean supervised row loss; the per-row cross entropy has a single implementation."""
    return _mean_rows(soft_cross_entropy_rows(logits, soft_target, mask), row_mask)
