"""Token-level (late-interaction) open-vocabulary allele matching.

A single pooled embedding per allele averages away the 1-2 SNPs that separate sibling alleles
(the measured accuracy ceiling). Late interaction keeps per-token detail: each read-segment
token takes its max cosine over a candidate germline's tokens, averaged over the segment, so
the one discriminating position survives instead of being washed out by the mean.

Concepts (clean re-implementation, not copied): MaxSim late interaction (ColBERT / GLIP
region-word), symmetric-contrastive temperature/logit_scale (open_clip). See sota/ATTRIBUTION.md.
"""
import math

import torch
import torch.nn as nn


def maxsim_scores(q: torch.Tensor, q_mask: torch.Tensor,
                  c: torch.Tensor, c_mask: torch.Tensor) -> torch.Tensor:
    """Late-interaction similarity of a batch of query segments against K candidates.

    q      (B, Sq, d)  query (read-segment) token reps, L2-normalized on d
    q_mask (B, Sq)     valid query tokens
    c      (K, Sc, d)  candidate (germline) token reps, L2-normalized on d
    c_mask (K, Sc)     valid candidate tokens
    -> (B, K) score = mean over valid query tokens of ( max over valid candidate tokens cos ).
    """
    sim = torch.einsum("bqd,kcd->bkqc", q, c)                       # (B,K,Sq,Sc) cosine
    sim = sim.masked_fill(~c_mask[None, :, None, :], float("-inf"))  # ignore candidate pads
    maxc = sim.max(dim=-1).values                                   # (B,K,Sq) best cand token
    qm = q_mask[:, None, :].to(maxc.dtype)                          # (B,1,Sq)
    maxc = maxc.masked_fill(qm == 0, 0.0)
    return (maxc * qm).sum(dim=-1) / qm.sum(dim=-1).clamp(min=1.0)  # (B,K)


class TokenMatch(nn.Module):
    """MaxSim late-interaction scores scaled by a learnable temperature (CLIP logit_scale).

    A genotype `candidate_mask` (K,) bool restricts scoring to the alleles in the caller's
    reference (disallowed -> -inf), the dynamic-genotype mechanism."""

    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100.0)

    def forward(self, q, q_mask, c, c_mask, candidate_mask=None) -> torch.Tensor:
        scores = maxsim_scores(q, q_mask, c, c_mask) * self.temperature
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask[None].to(scores.device), float("-inf"))
        return scores                                              # (B, K) logits

    def score_batched(self, q, q_mask, c, c_mask) -> torch.Tensor:
        """Per-batch candidate set (after top-k retrieval): c (B, K, Sc, d). -> (B, K) logits."""
        from .retrieval import maxsim_scores_batched
        return maxsim_scores_batched(q, q_mask, c, c_mask) * self.temperature


def contrastive_match_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-positive InfoNCE: -log( sum_{p in pos} e^{s_p} / sum_all e^{s_a} ). Treats
    indistinguishable alleles as an equivalence class (place mass anywhere in the true set).
    Hard negatives are supplied by WHICH candidates are scored (siblings included upstream),
    so the loss itself is generic. Rows with no positive contribute zero."""
    neg_inf = torch.finfo(scores.dtype).min
    has_pos = target.sum(dim=-1) > 0
    num = torch.logsumexp(scores.masked_fill(target <= 0, neg_inf), dim=-1)
    den = torch.logsumexp(scores, dim=-1)
    return torch.where(has_pos, den - num, torch.zeros_like(den)).mean()
