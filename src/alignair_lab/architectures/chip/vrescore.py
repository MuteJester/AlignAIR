"""The V re-score: the read's V span aligned with gaps to each of its top V rows, the alignment's score multiplied into
the V call (the design's stage 2, for V).

The V scan places each base at the column the reader gives it; an insertion or deletion shifts every column past it,
which a read's columns cannot always follow. Here the read's V span (the V start to the refined V end) is aligned to
each of the chip's ``TOP`` V alleles semi-globally (the span in full, the germline's ends free) in a band of ``BAND``
bases around the diagonal the conserved Cys fixes (read anchor a against the row's anchor base), base by base: a
match scores ``log(4(1-m))``, a mismatch ``log(4m/3)`` (m the read's mutation rate), a base inserted or deleted
``-GAP``, a read base past the row's germline ``OUTSIDE``. The scores, relative to the best, are added to those alleles'
log posterior; every other allele falls ``BEHIND`` (the candidates are the top ``TOP``). A read without its V anchor,
with a V span shorter than ``MIN_SPAN``, or whose candidates lack an annotated anchor keeps the scan's posterior. The
chip takes the product at inference only (the scan trains on its own posterior). No weights; no gradient.

On the lab's dev suites (chip-v1b, 20k steps; refiner, local D and bank orientation on) a parasail spike of the same
re-score took the V call from 98.9% to 99.3% (S5, V polymerase indels: 96.2% to 98.6%; S12, noise: 97.1% to 99.5%);
IgBLAST 99.4%.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from alignair.core.anchors import ANCHOR_LOGITS

BAND = 8                    # bases of net insertion or deletion the band holds
GAP = 4.0                   # the cost of a base inserted or deleted (about log(1 / 0.02))
OUTSIDE = -8.0              # a read base past the row's germline
TOP = 5                     # the candidates: the scan's top V alleles
MIN_SPAN = 20               # the shortest V span re-scored
BEHIND = -1e4               # how far behind the candidates every other allele falls
RATE = (0.01, 0.25)         # the mutation rate the ratios use, clamped
_NEG = -1e9


def banded_scores(tokens: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, anchors: torch.Tensor,
                  rows: torch.Tensor, row_anchors: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(B, K) each candidate row's banded semi-global alignment score against the read's span [start, end).

    ``tokens`` (B, L); ``starts``, ``ends``, ``anchors`` (B,): the span and the read position of the V anchor;
    ``rows`` (B, K, R): the candidates' ungapped germline tokens (0 past the end); ``row_anchors`` (B, K): their
    anchor bases; ``m`` (B, 1). Row base ``i + row_anchor - anchor + d`` is read base i's partner on band diagonal d.
    """
    width = 2 * BAND + 1
    band = torch.arange(width, device=tokens.device)
    offset = (row_anchors - anchors[:, None])[..., None] + band - BAND                    # (B, K, W)
    match = torch.log(4 * (1 - m))[..., None]
    mismatch = torch.log(4 * m / 3)[..., None]
    ramp = GAP * band.to(match.dtype)
    best = torch.zeros(offset.shape, dtype=match.dtype, device=tokens.device)          # free germline start
    for i in range(int(starts.min()), int(ends.max())):
        partner = offset + i
        inside = (partner >= 0) & (partner < rows.shape[-1])
        germline = rows.gather(-1, partner.clamp(0, rows.shape[-1] - 1))
        base = tokens[:, i][:, None, None]
        pair = torch.where(germline == base, match, mismatch)
        pair = torch.where(inside & (germline > 0), pair, torch.full_like(pair, OUTSIDE))
        pair = torch.where((base >= 1) & (base <= 4), pair, torch.zeros_like(pair))      # an N base: no evidence
        deleted = torch.cummax(best + ramp, -1).values - ramp                            # germline bases skipped
        inserted = F.pad(best[..., 1:], (0, 1), value=_NEG) - GAP                        # this read base unaligned
        step = torch.maximum(deleted + pair, inserted)
        best = torch.where(((i >= starts) & (i < ends))[:, None, None], step, best)
    return best.amax(-1)                                                                  # free germline end


@torch.no_grad()
def rescore_v(out: dict, tokens: torch.Tensor, bank) -> None:
    """Multiply the banded alignment of the read's V span into ``v_log_posterior`` and ``v_allele`` in ``out``."""
    log_posterior = out["v_log_posterior"]
    length = tokens.shape[1]
    top = log_posterior.topk(min(TOP, log_posterior.shape[-1]), -1).indices              # V rows are the alleles
    anchors = out[ANCHOR_LOGITS["v"]].argmax(-1)
    starts, ends = out["v_start_logits"].argmax(-1), out["v_end_logits"].argmax(-1)
    row_anchors = bank.anchor_base_v[top]
    usable = (anchors < length) & (ends - starts >= MIN_SPAN) & (row_anchors >= 0).all(-1)
    if not usable.any():
        return
    scores = banded_scores(tokens, torch.where(usable, starts, 0), torch.where(usable, ends, 0),
                           anchors.clamp_max(length - 1), bank.sequence_v[top], row_anchors,
                           out["mutation_rate"].clamp(*RATE))
    bonus = torch.full_like(log_posterior, BEHIND).scatter(1, top, scores - scores.amax(-1, keepdim=True))
    combined = torch.log_softmax(log_posterior + bonus, -1)
    out["v_log_posterior"] = torch.where(usable[:, None], combined, log_posterior)
    out["v_allele"] = torch.exp(out["v_log_posterior"] - out["v_log_posterior"].amax(-1, keepdim=True))
