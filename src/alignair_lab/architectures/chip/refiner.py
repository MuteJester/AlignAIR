"""The boundary refiner: the V end and the J start where the read stops following the called allele's germline.

The parser reads the regions from the read alone; the bank holds the germline. From the read's conserved anchor (the
anchor head's V Cys or J W/F codon start, read position a), read base i sits at base ``anchor_base(row) + (i - a)`` of
the called allele's ungapped germline (its top row). Each base within ``WINDOW`` of the anchor gives the
log-likelihood ratio of being that germline base, mutated at the read's mutation rate m, against being an N base:
``log(4(1-m))`` on a match, ``log(4m/3)`` on a mismatch, ``OUTSIDE`` where the allele has no base (past its germline
end, before its start). With
``S(x)`` the sum of the ratios of the bases before x, a V end at slot b gains ``S(b) - S(a)`` (the bases from the anchor
to b are V) and a J start at slot b gains ``S(a) - S(b)`` (the bases from b to the anchor are J), times
``EVIDENCE_WEIGHT``, on the parser's log-probability, its distribution the prior. An edge so moves outward over N bases that continue the germline and
inward over mutated edge bases, as the assay's boundary intervals allow. A read without the anchor, or whose called
allele has none annotated, keeps the parser's boundary. The evidence carries no gradient: the parser trains on its own
loss.
"""
from __future__ import annotations

import torch

from alignair.core.anchors import ANCHOR_LOGITS

from .parser import IMPOSSIBLE

WINDOW = 24               # bases from the anchor whose evidence counts
OUTSIDE = -8.0            # the ratio of a base the allele does not have
RATE = (0.01, 0.25)       # the mutation rate the ratios use, clamped
# The evidence's weight against the parser's log-probability. On the lab's dev suites (chip-v1b, 20k steps) the V end
# lands in the assay's interval for 88.7%, 90.7% and 91.9% of reads at 0.5, 1 and 2.
EVIDENCE_WEIGHT = 2.0
REFINED = {"v": "v_end_logits", "j": "j_start_logits"}


def _germline_tokens(bank, gene: str, rows: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    """(B, L) the called rows' germline tokens (1..4) at the given ungapped bases; 0 where the row has no base."""
    sequence = getattr(bank, f"sequence_{gene}")
    inside = (bases >= 0) & (bases < sequence.shape[1])
    return torch.where(inside, sequence[rows[:, None], bases.clamp(0, sequence.shape[1] - 1)], 0)


@torch.no_grad()
def _evidence(out: dict, tokens: torch.Tensor, mask: torch.Tensor, bank, gene: str):
    """``(evidence (B, L) per slot, refinable (B,))`` for the gene's refined boundary."""
    length = tokens.shape[1]
    anchor = out[ANCHOR_LOGITS[gene]].argmax(-1)
    rows = out[f"{gene}_log_posterior"].argmax(-1)              # V and J rows are the reference's alleles, in order
    anchor_base = getattr(bank, f"anchor_base_{gene}")[rows]
    refinable = (anchor < length) & (anchor_base >= 0)
    anchor = anchor.clamp_max(length - 1)
    position = torch.arange(length, device=tokens.device)
    germline = _germline_tokens(bank, gene, rows, anchor_base[:, None] + position - anchor[:, None])
    m = out["mutation_rate"].clamp(*RATE)
    ratio = torch.where(tokens == germline, torch.log(4 * (1 - m)), torch.log(4 * m / 3))
    ratio = torch.where(germline == 0, torch.full_like(ratio, OUTSIDE), ratio)
    counted = mask & (tokens >= 1) & (tokens <= 4) & ((position - anchor[:, None]).abs() <= WINDOW)
    ratio = torch.where(counted, ratio, torch.zeros_like(ratio))
    before = torch.cat([torch.zeros_like(ratio[:, :1]), ratio.cumsum(-1)], -1)          # S(x), x = 0..L
    at_anchor = before.gather(1, anchor[:, None])
    evidence = before[:, :length] - at_anchor if gene == "v" else at_anchor - before[:, :length]
    return evidence, refinable


def refine_boundaries(out: dict, tokens: torch.Tensor, mask: torch.Tensor, end_mask: torch.Tensor, bank) -> None:
    """Refine ``v_end_logits`` and ``j_start_logits`` in ``out`` (and their expectations ``v_end``, ``j_start``)."""
    for gene, key in REFINED.items():
        evidence, refinable = _evidence(out, tokens, mask, bank, gene)
        slots = end_mask if key.endswith("end_logits") else mask                # a slot no read has stays as it is
        out[key] = torch.where(refinable[:, None] & slots, out[key] + EVIDENCE_WEIGHT * evidence, out[key])
        probs = out[key].masked_fill(~slots, IMPOSSIBLE).softmax(-1)
        out[key.removesuffix("_logits")] = (probs * torch.arange(probs.shape[-1], device=probs.device)).sum(
            -1, keepdim=True)
