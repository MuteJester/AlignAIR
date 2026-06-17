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
        # Clamp the temperature to a floor of 0.04 (and ceil 1.0): query/candidates
        # are unit-normalised, so cosine in [-1,1] and scores in [-25,25]. Without a
        # floor the learnable temperature collapses toward 0, blowing scores (and the
        # InfoNCE loss on a wrong positive) up to ~200 and destabilising training.
        temp = self.log_temp.exp().clamp(0.04, 1.0)
        scores = (query @ candidates.t()) / temp
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.unsqueeze(0), float("-inf"))
        return scores


def contrastive_match_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-positive (set) InfoNCE: ``-log( sum_{p in pos} e^{s_p} / sum_{all} e^{s_a} )``.

    This treats genuinely indistinguishable alleles as an EQUIVALENCE CLASS — the
    model only needs to place its mass somewhere in the true set, not split it
    equally across positives (which the old per-positive average forced, making
    co-listed alleles compete in the softmax). Aligns with the top-1-in-set metric.
    Rows with no positive (e.g. masked inverted-D) contribute zero."""
    neg_inf = torch.finfo(scores.dtype).min
    has_pos = target.sum(dim=-1) > 0                          # (B,)
    pos_scores = scores.masked_fill(target <= 0, neg_inf)     # keep positives only
    num = torch.logsumexp(pos_scores, dim=-1)                 # log-mass on the true set
    den = torch.logsumexp(scores, dim=-1)                     # log-mass over all
    row_loss = torch.where(has_pos, den - num, torch.zeros_like(den))
    return row_loss.mean()


def multilabel_match_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-label BCE over candidates. ``target`` is a (B, K) multi-hot of true alleles.

    Columns with -inf scores (genotype-masked) are ignored so they never contribute."""
    finite = torch.isfinite(scores)
    safe_scores = torch.where(finite, scores, torch.zeros_like(scores))
    per_elem = F.binary_cross_entropy_with_logits(safe_scores, target, reduction="none")
    per_elem = per_elem * finite.to(per_elem.dtype)
    return per_elem.sum() / finite.to(per_elem.dtype).sum().clamp(min=1.0)
