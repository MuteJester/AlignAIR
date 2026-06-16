"""Allele-embedding matching head: cosine similarity (temperature-scaled) ->
multi-label allele scores, with optional genotype candidate masking."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlleleMatchingHead(nn.Module):
    """Score normalized queries (B, d) against normalized candidate embeddings (K, d).

    Returns (B, K) logits = cosine_sim / temperature. A genotype is a (K,) bool mask
    of allowed candidates (disallowed -> -inf, i.e. sigmoid 0).
    """

    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    def forward(self, query: torch.Tensor, candidates: torch.Tensor,
                candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = (query @ candidates.t()) / self.log_temp.exp().clamp(min=1e-4)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.unsqueeze(0), float("-inf"))
        return scores


def contrastive_match_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-positive InfoNCE over candidates: push every true allele's score above
    all others via a softmax-CE, averaged over the positive set per row. Robust to
    large candidate sets where multi-label BCE collapses to 'predict nothing'."""
    log_probs = F.log_softmax(scores, dim=-1)            # (B, K)
    pos_per_row = target.sum(dim=-1).clamp(min=1.0)      # (B,)
    row_loss = -(log_probs * target).sum(dim=-1) / pos_per_row
    return row_loss.mean()


def multilabel_match_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-label BCE over candidates. ``target`` is a (B, K) multi-hot of true alleles.

    Columns with -inf scores (genotype-masked) are ignored so they never contribute."""
    finite = torch.isfinite(scores)
    safe_scores = torch.where(finite, scores, torch.zeros_like(scores))
    per_elem = F.binary_cross_entropy_with_logits(safe_scores, target, reduction="none")
    per_elem = per_elem * finite.to(per_elem.dtype)
    return per_elem.sum() / finite.to(per_elem.dtype).sum().clamp(min=1.0)
