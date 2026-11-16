"""The boundary refiner: the V end and the J start where the read stops following the called allele's germline.

The parser reads the regions from the read alone; the bank holds the germline. From the read's conserved anchor (the
anchor head's V Cys or J W/F codon start, read position a), the read is aligned against the called allele's ungapped
germline (its top row) from its anchor base ``anchor_base(row)``, 3'-ward and 5'-ward over ``WINDOW`` bases. A read base
paired with a germline base gives the log-likelihood ratio of being that germline base, mutated at the read's mutation
rate m, against being an N base: ``log(4(1-m))`` on a match, ``log(4m/3)`` on a mismatch, ``OUTSIDE`` where the allele
has no base (past its germline end, before its start); a base inserted into the read or deleted from it costs ``GAP``,
within ``BAND`` diagonals of the anchor's. With ``F(k)`` the best score of the k read bases from the anchor 3'-ward and
``R(k)`` of the k bases 5' of it (the germline's far end free), a V end at slot b gains ``F(b - a)`` for b >= a (the
bases from the anchor to b are V) and ``-R(a - b)`` for b < a, and a J start at slot b gains ``R(a - b)`` for b <= a
(the bases from b to the anchor are J) and ``-F(b - a)`` for b > a, times ``EVIDENCE_WEIGHT``, on the parser's
log-probability, its distribution the prior. An edge so moves outward over N bases that continue the germline and inward
over mutated edge bases, the freedom the training targets give a boundary (``data.chip_inputs``). A read without the
anchor, or whose called allele has none annotated, keeps the parser's boundary. The evidence carries no gradient: the
parser trains on its own loss.

The gaps let a boundary pass an indel. Without them (``GAP`` -inf) the comparison is ungapped, read base i <->
germline base ``anchor_base + (i - a)``: an indel between the anchor and a boundary stops the boundary at the indel,
and one just before the anchor makes the shifted bases before it look non-V, pulling the V end back past the anchor.
"""
from __future__ import annotations

import torch

from ...core.anchors import ANCHOR_LOGITS
from ...core.tokens import is_base
from ..parser import expected_slot
from .ratios import OUTSIDE, RATE, pair_ratios

WINDOW = 24               # bases from the anchor whose evidence counts
BAND = 3                  # the most net indels the alignment follows between the anchor and a boundary
# The ratio of a base inserted into the read or deleted from it: of -2, -3, -4 and -6, the one that places the edges
# of reads with polymerase indels best without costing any other kind of read.
GAP = -6.0
_UNREACHED = -1e4         # the score of a diagonal no path has reached
# The evidence's weight against the parser's log-probability: of 0.5, 1 and 2, the one that places the V end best.
EVIDENCE_WEIGHT = 2.0
REFINED = {"v": "v_end_logits", "j": "j_start_logits"}


def _germline_tokens(bank, gene: str, rows: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    """(B, L) the called rows' germline tokens (1..4) at the given ungapped bases; 0 where the row has no base."""
    sequence = getattr(bank, f"sequence_{gene}")
    inside = (bases >= 0) & (bases < sequence.shape[1])
    return torch.where(inside, sequence[rows[:, None], bases.clamp(0, sequence.shape[1] - 1)], 0)


def _path_scores(read: torch.Tensor, germline: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(N, K + 1) the best score of the first k = 0..K read bases walked from the anchor against the germline walked
    from its anchor base, indels within ``BAND`` diagonals, the germline's far end free. ``read`` (N, K) the walked read
    bases, ``germline`` (N, K, 2 * BAND + 1) the germline base each diagonal pairs with each read base (0: none)."""
    match, mismatch = pair_ratios(m)
    ratio = torch.where(read[:, :, None] == germline, match[:, :, None], mismatch[:, :, None])
    ratio = torch.where(germline == 0, OUTSIDE, ratio)
    ratio = torch.where(is_base(read)[:, :, None], ratio, 0.)  # an N base says nothing
    # an inserted read base costs GAP while the germline goes on, OUTSIDE once it has ended (none off the band's edge)
    inserted = torch.where(germline[:, :, 1:] != 0, GAP, OUTSIDE)
    inserted = torch.cat([inserted, torch.zeros_like(inserted[:, :, :1])], 2)
    ladder = torch.arange(germline.shape[2], device=read.device, dtype=ratio.dtype) * GAP
    edge = torch.full((read.shape[0], 1), _UNREACHED, device=read.device, dtype=ratio.dtype)
    score = torch.full((read.shape[0], germline.shape[2]), _UNREACHED, device=read.device, dtype=ratio.dtype)
    score[:, BAND] = 0.
    score = ladder + torch.cummax(score - ladder, 1).values               # germline bases deleted at the anchor
    history = [score]
    for k in range(read.shape[1]):
        step = torch.maximum(score + ratio[:, k], torch.cat([score[:, 1:], edge], 1) + inserted[:, k])
        score = ladder + torch.cummax(step - ladder, 1).values            # then germline bases deleted after it
        history.append(score)
    return torch.stack(history, 1).amax(-1)


@torch.no_grad()
def _evidence(out: dict, tokens: torch.Tensor, mask: torch.Tensor, bank) -> dict:
    """``{gene: (evidence (B, L) per slot, refinable (B,))}`` for each refined boundary; the four walks (V and J, 3'-
    and 5'-ward of the anchor) are aligned as one batch."""
    length = tokens.shape[1]
    read = torch.where(mask, tokens, 0)
    k = torch.arange(WINDOW + 1, device=tokens.device)
    diagonal = k[:, None] + torch.arange(-BAND, BAND + 1, device=tokens.device)              # (K, width): k + d
    walks, germlines, anchors = [], [], {}
    for gene in REFINED:
        anchor = out[ANCHOR_LOGITS[gene]].argmax(-1)
        rows = out[f"{gene}_log_posterior"].argmax(-1)          # V and J rows are the reference's alleles, in order
        anchor_base = getattr(bank, f"anchor_base_{gene}")[rows]
        anchors[gene] = (anchor.clamp_max(length - 1), (anchor < length) & (anchor_base >= 0))
        anchor = anchors[gene][0]
        # 3'-ward: read base a + k against germline base A + k + d; 5'-ward: a - 1 - k against A - 1 - k - d
        for positions, bases in ((anchor[:, None] + k, anchor_base[:, None, None] + diagonal),
                                 (anchor[:, None] - 1 - k, anchor_base[:, None, None] - 1 - diagonal)):
            inside = (positions >= 0) & (positions < length)
            walks.append(torch.where(inside, read.gather(1, positions.clamp(0, length - 1)), 0))
            germlines.append(_germline_tokens(bank, gene, rows, bases.flatten(1)).view(bases.shape))
    m = out["mutation_rate"].clamp(*RATE)
    scores = _path_scores(torch.cat(walks), torch.cat(germlines), m.repeat(len(walks), 1)).chunk(len(walks))
    evidence = {}
    for (gene, (anchor, refinable)), ahead, behind in zip(anchors.items(), scores[0::2], scores[1::2]):
        # the bases within WINDOW of the anchor count, the anchor's own on its 3' side
        offset = torch.arange(length, device=tokens.device) - anchor[:, None]
        ahead = ahead.gather(1, offset.clamp(0, WINDOW + 1))
        behind = behind.gather(1, (-offset).clamp(0, WINDOW))
        if gene == "v":
            # a V end at b: the bases from the anchor to b are V, those from b to the anchor are not
            evidence[gene] = torch.where(offset >= 0, ahead, -behind), refinable
        else:
            # a J start at b: the bases from b to the anchor are J, those from the anchor to b are not
            evidence[gene] = torch.where(offset <= 0, behind, -ahead), refinable
    return evidence


def refine_boundaries(out: dict, tokens: torch.Tensor, mask: torch.Tensor, end_mask: torch.Tensor, bank) -> None:
    """Refine ``v_end_logits`` and ``j_start_logits`` in ``out`` (and their expectations ``v_end``, ``j_start``)."""
    evidences = _evidence(out, tokens, mask, bank)
    for gene, key in REFINED.items():
        evidence, refinable = evidences[gene]
        slots = end_mask if key.endswith("end_logits") else mask                # a slot no read has stays as it is
        out[key] = torch.where(refinable[:, None] & slots, out[key] + EVIDENCE_WEIGHT * evidence, out[key])
        out[key.removesuffix("_logits")] = expected_slot(out[key], slots)
