"""The V re-score: the read's V span aligned with gaps to each of its top V rows, the alignment's score multiplied into
the V call.

The V scan places each base at the column the reader gives it; an insertion or deletion shifts every column past it,
which a read's columns cannot always follow. Here the read's V span (the V start to the refined V end) is aligned to
each of the chip's ``TOP`` V alleles semi-globally (the span in full, the germline's ends free) in a band of ``BAND``
bases around the diagonal the conserved Cys fixes (read anchor a against the row's anchor base), base by base: a
match scores ``log(4(1-m))``, a mismatch ``log(4m/3)`` (m the read's mutation rate), a base inserted or deleted
``-GAP_COST``, a read base past the row's germline ``OUTSIDE``. The scores, relative to the best, are added to those
alleles' log posterior; every other allele falls ``BEHIND`` (the candidates are the top ``TOP``). A read without its V
anchor, with a V span shorter than ``MIN_SPAN``, or whose candidates lack an annotated anchor keeps the scan's
posterior. The chip takes the product at inference only (the scan trains on its own posterior). No weights; no
gradient.

The band runs in a compiled kernel (``_vband_cy``) when it is built, else in torch (``VBAND_BACKEND`` reports which);
the two give the same scores bit for bit.
"""
from __future__ import annotations

import numpy as np
import torch

from ...core.anchors import ANCHOR_LOGITS
from ...core.tokens import is_base
from .ratios import OUTSIDE, RATE, pair_ratios

BAND = 8                    # bases of net insertion or deletion the band holds
GAP_COST = 4.0              # the cost of a base inserted or deleted (about log(1 / 0.02))
TOP = 5                     # the candidates: the scan's top V alleles
MIN_SPAN = 20               # the shortest V span re-scored
BEHIND = -1e4               # how far behind the candidates every other allele falls
_NEG = -1e9

try:                                                        # optional compiled band (the torch band's scores)
    from ._vband_cy import band_scores as _compiled_band
    VBAND_BACKEND = "cython"
except (ImportError, OSError):                              # pragma: no cover - optional compiled kernel
    _compiled_band = None
    VBAND_BACKEND = "python"


def banded_scores(tokens: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, anchors: torch.Tensor,
                  rows: torch.Tensor, row_anchors: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(B, K) each candidate row's banded semi-global alignment score against the read's span [start, end).

    ``tokens`` (B, L); ``starts``, ``ends``, ``anchors`` (B,): the span and the read position of the V anchor;
    ``rows`` (B, K, R): the candidates' ungapped germline tokens (0 past the end); ``row_anchors`` (B, K): their
    anchor bases; ``m`` (B, 1). Row base ``i + row_anchor - anchor + d`` is read base i's partner on band diagonal d.
    The compiled kernel runs on the CPU whatever the reads' device (its inputs are a few small integer tensors); the
    scores come back on the reads' device.
    """
    if _compiled_band is None or m.dtype != torch.float32 or rows.shape[0] == 0:
        return _torch_banded_scores(tokens, starts, ends, anchors, rows, row_anchors, m)
    match, mismatch = pair_ratios(m)

    def host(x, dtype):
        return x.detach().to("cpu", dtype).contiguous().numpy()

    starts_, ends_ = host(starts, torch.int64), host(ends, torch.int64)
    if starts_.min() < 0 or ends_.max() > tokens.shape[1]:
        raise IndexError(f"a V span [{starts_.min()}, {ends_.max()}) runs past the read's {tokens.shape[1]} bases")
    out = np.empty(rows.shape[:2], dtype=np.float32)
    _compiled_band(host(tokens, torch.int64), starts_, ends_, host(anchors, torch.int64), host(rows, torch.int64),
                   host(row_anchors, torch.int64), host(match.reshape(-1), torch.float32),
                   host(mismatch.reshape(-1), torch.float32), BAND, GAP_COST, OUTSIDE, _NEG, out)
    return torch.from_numpy(out).to(tokens.device)


def _torch_banded_scores(tokens: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, anchors: torch.Tensor,
                         rows: torch.Tensor, row_anchors: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``banded_scores`` in torch: every position's pair scores at once, then the band's recursion (``band_dp``)."""
    width = 2 * BAND + 1
    band = torch.arange(width, device=tokens.device)
    first, last = int(starts.min()), int(ends.max())
    positions = torch.arange(first, max(last, first + 1), device=tokens.device)          # (T,)
    # every position's pair score at once: (B, K, T, W)
    partner = ((row_anchors - anchors[:, None])[..., None, None] + positions[:, None] + band - BAND)
    inside = (partner >= 0) & (partner < rows.shape[-1])
    batch, candidates = rows.shape[:2]
    germline = rows.gather(-1, partner.clamp(0, rows.shape[-1] - 1).reshape(batch, candidates, -1)).view(
        partner.shape)
    base = tokens[:, positions][:, None, :, None]
    match, mismatch = pair_ratios(m)
    pair = torch.where(germline == base, match[..., None, None], mismatch[..., None, None])
    pair = torch.where(inside & (germline > 0), pair, torch.full_like(pair, OUTSIDE))
    pair = torch.where(is_base(base), pair, torch.zeros_like(pair))  # an N base: no evidence
    active = ((positions >= starts[:, None]) & (positions < ends[:, None]))[:, None, :, None]   # (B, 1, T, 1)
    return band_dp(pair, active)


def band_dp(pair: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """(B, K) the band's best score from its pair scores ``pair`` (B, K, T, W) over the read bases ``active``
    (B, 1, T, 1) take part in. ``best`` (B, K, W) over the band's diagonals starts at 0 (a free germline start) and
    ends at its max (a free end); at each base, diagonal w takes the best of a diagonal at or left of it (germline
    bases skipped, ``GAP_COST`` each) plus the pair score, or diagonal w + 1 less ``GAP_COST`` (the base unaligned).
    Kept as ``best + GAP_COST * w``, the deletion is a plain running max and the insertion ``2 * GAP_COST`` down, so
    each base is a few ops on (B, K, W) tensors (a base's cost on a CPU is mostly its ops' fixed overhead)."""
    ramp = GAP_COST * torch.arange(pair.shape[-1], dtype=pair.dtype, device=pair.device)
    ramped = ramp.expand(*pair.shape[:2], -1).clone()
    inserted = torch.full_like(ramped, _NEG)
    for pair_t, active_t in zip(pair.unbind(2), active.unbind(2)):
        inserted[..., :-1] = ramped[..., 1:] - 2 * GAP_COST
        ramped = torch.where(active_t, torch.maximum(torch.cummax(ramped, -1).values + pair_t, inserted), ramped)
    return (ramped - ramp).amax(-1)


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
