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

from .scan import allele_log_posterior

FLANK = 6                  # bases on each side of the V end and J start the junction keeps
CROP = 96                  # the junction's most bases
CHUNK = 32                 # reads aligned at a time (the (reads, rows, offsets, row width) ratios held at once)
RATE = (0.01, 0.25)        # the mutation rate the ratios use, clamped
TEMPERATURE = 1.0          # the scores' temperature in the alleles' log posterior


def junction(tokens: torch.Tensor, mask: torch.Tensor, v_end: torch.Tensor, j_start: torch.Tensor) -> torch.Tensor:
    """(B, CROP) the read's bases from ``v_end - FLANK`` to ``j_start + FLANK`` (A/C/G/T tokens; 0 elsewhere)."""
    length = tokens.shape[1]
    start = (v_end - FLANK).clamp(0, length - 1)
    stop = torch.minimum(j_start + FLANK, mask.sum(1))
    position = start[:, None] + torch.arange(CROP, device=tokens.device)
    crop = tokens.gather(1, position.clamp_max(length - 1))
    return torch.where((position < stop[:, None]) & (crop >= 1) & (crop <= 4), crop, 0)


def local_scores(crop: torch.Tensor, rows: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(B, R) each row's best ungapped local alignment score against the crop; ``m`` (B, 1) the mutation rate."""
    width = rows.shape[1]
    windows = F.pad(crop, (width - 1, width - 1)).unfold(1, width, 1)                       # (B, offsets, W)
    match = torch.log(4 * (1 - m))[..., None, None]
    mismatch = torch.log(4 * m / 3)[..., None, None]
    scores = []
    for first in range(0, crop.shape[0], CHUNK):
        window = windows[first:first + CHUNK, None]                                         # (b, 1, offsets, W)
        both = (window > 0) & (rows[None, :, None] > 0)
        ratio = torch.where(both & (window == rows[None, :, None]), match[first:first + CHUNK],
                            torch.where(both, mismatch[first:first + CHUNK], torch.zeros_like(match[:1])))
        prefix = F.pad(ratio.cumsum(-1), (1, 0))
        scores.append((prefix - prefix.cummin(-1).values).amax(-1).amax(-1))
    return torch.cat(scores)


@torch.no_grad()
def local_d_log_posterior(out: dict, tokens: torch.Tensor, mask: torch.Tensor, bank, alleles: int) -> torch.Tensor:
    """(B, alleles) the D alleles' log posterior from the local alignment of the junction the outputs place."""
    crop = junction(tokens, mask, out["v_end_logits"].argmax(-1), out["j_start_logits"].argmax(-1))
    scores = local_scores(crop, bank.sequence_d, out["mutation_rate"].clamp(*RATE))
    return allele_log_posterior(scores / TEMPERATURE, bank.row_allele_d, alleles)
