"""The local D aligner: the D call from the read's junction aligned base by base to every D row.

The D scan places each base at its column from the reader's coordinates, which a short D (trimmed at both ends) or an
inverted D leaves it little to go on. Here the junction (the read from ``FLANK`` bases before the V end to ``FLANK``
after the J start, at most ``CROP`` bases) slides along every D row, forward and reverse complement, without gaps;
along each diagonal each base pair gives ``log(4(1-m))`` on a match and ``log(4m/3)`` on a mismatch (its ratio against
a random base, m the read's mutation rate), and the diagonal scores its best contiguous stretch, so a row is never
charged for the bases a trim removed. A row scores its best diagonal; the alleles' log posterior from those scores (an
allele's two rows summed, at ``TEMPERATURE``) is added to the D scan's (a product of the two), which keeps the scan's
learned preferences where the read is long and lets the alignment decide where it is short. No weights; no gradient.
The chip takes the product at inference only: the scan trains on its own posterior, a calibrated expert of its own
(the alignment already decides an easy read, which would leave the scan nothing to learn from it).

On the lab's dev suites (chip-v1b, 20k steps; refiner and bank orientation on) the D call is right for 95.0% of reads
with the scan alone, 91.3% with the alignment alone, 96.9% with their product at temperature 1 (S6, D trimmed at
both ends or inverted: 61.0%, 94.9%, 94.9%) and less at temperature 2 (S6 trim 9 and 12: 71% and 64% against 89%
and 91%); IgBLAST 90.0% (S6 74.4%).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ..network.scan import allele_log_posterior

FLANK = 6                  # bases on each side of the V end and J start the junction keeps
CROP = 96                  # the junction's most bases
RATE = (0.01, 0.25)        # the mutation rate the ratios use, clamped
TEMPERATURE = 1.0          # the scores' temperature in the alleles' log posterior


def junction_bounds(mask: torch.Tensor, v_end: torch.Tensor, j_start: torch.Tensor) -> tuple:
    """``(start, stop)`` (B,): the read positions from ``v_end - FLANK`` to ``j_start + FLANK``, within the read."""
    start = (v_end - FLANK).clamp(0, mask.shape[1] - 1)
    return start, torch.minimum(j_start + FLANK, mask.sum(1))


def junction(tokens: torch.Tensor, mask: torch.Tensor, v_end: torch.Tensor, j_start: torch.Tensor) -> torch.Tensor:
    """(B, width) the read's bases from ``v_end - FLANK`` to ``j_start + FLANK`` (A/C/G/T tokens; 0 elsewhere): the
    width is the batch's longest junction, at most ``CROP``."""
    length = tokens.shape[1]
    start, stop = junction_bounds(mask, v_end, j_start)
    width = int((stop - start).clamp(1, CROP).max())
    position = start[:, None] + torch.arange(width, device=tokens.device)
    crop = tokens.gather(1, position.clamp_max(length - 1))
    return torch.where((position < stop[:, None]) & (crop >= 1) & (crop <= 4), crop, 0)


def local_scores(crop: torch.Tensor, rows: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(B, R) each row's best ungapped local alignment score against the crop; ``m`` (B, 1) the mutation rate.
    Kadane along the row: every (read, diagonal, row) carries its best stretch ending at the row base reached. A
    pair's ratio is one small product: the read base as (mismatch if a base, (match - mismatch) x its one-hot), the
    row base as (1 if a base, its one-hot), so a match gives match, a mismatch mismatch and a pad 0, for every
    (diagonal, row) of a row base at once."""
    width = rows.shape[1]
    match, mismatch = torch.log(4 * (1 - m)), torch.log(4 * m / 3)                         # (B, 1)
    padded = F.pad(crop, (width - 1, width - 1))                                           # (B, offsets + W - 1)
    read = torch.cat([mismatch[..., None] * (padded > 0)[..., None],
                      (match - mismatch)[..., None] * F.one_hot(padded, 6)[..., 1:]], -1).to(match.dtype)
    germline = torch.cat([(rows > 0)[..., None], F.one_hot(rows, 6)[..., 1:]], -1).to(match.dtype)   # (R, W, 6)
    offsets = padded.shape[1] - width + 1
    running = torch.zeros(crop.shape[0], offsets, rows.shape[0], dtype=match.dtype, device=crop.device)
    best = torch.zeros_like(running)
    for k in range(width):
        running = (running + read[:, k:k + offsets] @ germline[:, k].t()).clamp_min_(0.)  # (B, O, R)
        best = torch.maximum(best, running)
    return best.amax(1)


@torch.no_grad()
def local_d_log_posterior(out: dict, tokens: torch.Tensor, mask: torch.Tensor, bank, alleles: int) -> torch.Tensor:
    """(B, alleles) the D alleles' log posterior from the local alignment of the junction the outputs place."""
    crop = junction(tokens, mask, out["v_end_logits"].argmax(-1), out["j_start_logits"].argmax(-1))
    scores = local_scores(crop, bank.sequence_d, out["mutation_rate"].clamp(*RATE))
    return allele_log_posterior(scores / TEMPERATURE, bank.row_allele_d, alleles)
