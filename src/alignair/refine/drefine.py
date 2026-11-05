"""The D refiner: the D start and end where the read stops following the called D allele's germline.

The parser places the D from the read alone, which a narrow trunk does less well than a wide one; the bank holds the
germline. The junction (``dlocal.junction``) slides along the called allele's two rows (forward and reverse
complement) without gaps, as the local D aligner slides it along every row, and the row and diagonal whose best
contiguous stretch scores highest line the germline up with the read; an inverted D so finds its reverse complement
row. Along that diagonal each read base gives the log-likelihood ratio of being its germline base, mutated at the
read's mutation rate m, against being an N base: ``log(4(1-m))`` on a match, ``log(4m/3)`` on a mismatch, ``OUTSIDE``
where the diagonal pairs it with no germline base (a D has no bases past its germline); an N base says nothing. With
S(b) the summed ratios of the read bases before slot b, a D from slot s to slot e > s scores
``start(s) + end(e) + EVIDENCE_WEIGHT * (S(e) - S(s))``, the parser's boundary log-probabilities the prior, and each
boundary's refined logits are that score summed over the other boundary (a start over the ends after it, an end over
the starts before it): one edge alone cannot run past the other, where S keeps falling. The D lies between the V end
and the J start the outputs place (the product's segments order them so, and would move the J start to a D end past
it), so a germline that runs on into the V or the J stops there. An edge so moves outward over N bases that continue
the germline and inward over mutated edge bases, as the assay's boundary intervals allow. A read whose junction aligns
no better than ``MIN_SCORE`` to its called allele, or whose V end is not before its J start, keeps the parser's D. No
weights; no gradient; at inference, after the D call.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .dlocal import RATE, junction, junction_bounds
from ..network.parser import IMPOSSIBLE

OUTSIDE = -8.0             # the ratio of a read base the diagonal pairs with no germline base
MIN_SCORE = 0.0            # the best stretch a read's junction needs on its called allele for the D to be refined
# The evidence's weight against the parser's log-probability. On the lab's dev suites (the width screens, 20k steps;
# IgBLAST 74.1% and 75.2%) the D start and end land in the assay's intervals for 81.5% and 81.0% of reads without the
# refiner on the 128-channel trunk, 86.6/87.2, 86.9/87.2, 86.8/87.0 and 86.6/86.1 at 1.5, 2, 3 and 4; on the
# 64-channel trunk 76.2/69.7 without, 86.1/85.7, 86.3/86.0, 85.8/85.2 and 85.4/84.8 (S6, a D trimmed at both ends or
# inverted: 42.6/42.1 without, 80.5/80.5 at 2; IgBLAST 72.3/69.7). The V and J calls and edges do not move.
EVIDENCE_WEIGHT = 2.0


def called_rows(log_posterior: torch.Tensor, row_allele: torch.Tensor) -> torch.Tensor:
    """(B, 2) the bank rows of each read's called allele: its forward row, then its reverse complement's."""
    called = (row_allele[None, :] == log_posterior.argmax(-1)[:, None]).nonzero()
    return called[:, 1].view(log_posterior.shape[0], 2)


def best_diagonal(crop: torch.Tensor, rows: torch.Tensor, m: torch.Tensor) -> tuple:
    """``(score, row, shift)`` (B,) of the read's ``rows`` (B, K, W) the one and the diagonal whose best contiguous
    stretch against ``crop`` (B, C) scores highest: its score, its index among the K and the shift that pairs crop
    base i with row base i - shift."""
    width = rows.shape[-1]
    windows = F.pad(crop, (width - 1, width - 1)).unfold(1, width, 1)     # window o pairs crop o + k - (W - 1) with k
    read, germline = windows[:, None], rows[:, :, None]                     # (B, 1, O, W), (B, K, 1, W)
    both = (read > 0) & (germline > 0)
    match, mismatch = torch.log(4 * (1 - m))[..., None, None], torch.log(4 * m / 3)[..., None, None]
    ratio = torch.where(both & (read == germline), match, torch.where(both, mismatch, 0.))
    total = F.pad(ratio.cumsum(-1), (1, 0))
    stretch = (total - total.cummin(-1).values).amax(-1)                    # (B, K, O): Kadane's best stretch
    score, flat = stretch.flatten(1).max(-1)
    return score, flat // stretch.shape[2], flat % stretch.shape[2] - (width - 1)


def evidence(tokens: torch.Tensor, mask: torch.Tensor, germline: torch.Tensor, shift: torch.Tensor,
             m: torch.Tensor) -> torch.Tensor:
    """(B, L) S(b), the summed ratios of the read bases before slot b, read base p paired with germline base
    ``p - shift`` of ``germline`` (B, W)."""
    length, width = tokens.shape[1], germline.shape[1]
    base = torch.arange(length, device=tokens.device) - shift[:, None]
    paired = torch.where((base >= 0) & (base < width), germline.gather(1, base.clamp(0, width - 1)), 0)
    ratio = torch.where(paired > 0, torch.where(tokens == paired, torch.log(4 * (1 - m)), torch.log(4 * m / 3)),
                        OUTSIDE)
    ratio = torch.where(mask & (tokens >= 1) & (tokens <= 4), ratio, 0.)
    return F.pad(ratio.cumsum(1), (1, 0))[:, :length]


@torch.no_grad()
def refine_d_boundaries(out: dict, tokens: torch.Tensor, mask: torch.Tensor, end_mask: torch.Tensor, bank) -> None:
    """Refine ``d_start_logits`` and ``d_end_logits`` in ``out`` (and their expectations ``d_start``, ``d_end``) on
    the germline of the D call ``d_log_posterior``."""
    v_end, j_start = out["v_end_logits"].argmax(-1), out["j_start_logits"].argmax(-1)
    crop, (start, _) = junction(tokens, mask, v_end, j_start), junction_bounds(mask, v_end, j_start)
    rows = called_rows(out["d_log_posterior"], bank.row_allele_d)
    germlines = bank.sequence_d[rows]                                       # (B, 2, W)
    m = out["mutation_rate"].clamp(*RATE)
    score, row, shift = best_diagonal(crop, germlines, m)
    germline = germlines.gather(1, row[:, None, None].expand(-1, 1, germlines.shape[2]))[:, 0]
    slot = torch.arange(tokens.shape[1], device=tokens.device)
    after_v, before_j = mask & (slot >= v_end[:, None]), end_mask & (slot <= j_start[:, None])
    # only the bases between the V end and the J start count: the D lies there, and S stays of the junction's size
    s = evidence(tokens, after_v & (slot < j_start[:, None]), germline, start + shift, m)
    starts = (out["d_start_logits"] - EVIDENCE_WEIGHT * s).masked_fill(~after_v, -torch.inf)
    ends = (out["d_end_logits"] + EVIDENCE_WEIGHT * s).masked_fill(~before_j, -torch.inf)
    later = F.pad(torch.logcumsumexp(ends.flip(-1), -1).flip(-1)[:, 1:], (0, 1), value=-torch.inf)  # ends e > s
    earlier = F.pad(torch.logcumsumexp(starts, -1)[:, :-1], (1, 0), value=-torch.inf)               # starts s < e
    refinable = ((score > MIN_SCORE) & (v_end < j_start))[:, None]
    for key, refined, slots in (("d_start_logits", starts + later, mask), ("d_end_logits", ends + earlier, end_mask)):
        out[key] = torch.where(refinable, refined.clamp_min(IMPOSSIBLE), out[key])   # a slot no D uses: impossible
        probs = out[key].masked_fill(~slots, IMPOSSIBLE).softmax(-1)
        out[key.removesuffix("_logits")] = (probs * torch.arange(probs.shape[-1], device=probs.device)).sum(
            -1, keepdim=True)
