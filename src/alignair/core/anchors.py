"""The conserved V and J anchors a model's anchor heads locate: the one
definition of their codon length, their genes and their keys.

An anchor is the read position of the first base of a conserved codon: V's Cys, J's Trp/Phe. The junction runs from
the V anchor to the end of the J anchor's codon (``predict.junction``). A model scores each read position as an anchor,
plus a null slot ("not observed"); its targets (``data``), loss terms and weights (``tasks``) and records (``predict``)
are keyed by ``ANCHOR_KEYS``, its outputs by ``ANCHOR_LOGITS``.
"""
from __future__ import annotations

import torch

CODON = 3                                              # bases in an anchor codon
ANCHOR_GENES = ("v", "j")                              # the genes with a conserved anchor, in loss-weight order
ANCHOR_KEYS = {gene: f"{gene}_anchor" for gene in ANCHOR_GENES}            # targets, loss terms, record fields
ANCHOR_LOGITS = {gene: f"{gene}_anchor_logits" for gene in ANCHOR_GENES}   # model outputs (B, L + 1)


def codon_starts(read: torch.Tensor) -> torch.Tensor:
    """The slots an anchor can take: (B, L) bool, True at read position k when k, k + 1 and k + 2 are all read
    positions of ``read`` (B, L bool), so a whole codon fits there. The anchor loss and the decoder use them."""
    starts = read.clone()
    for shift in range(1, CODON):
        starts[:, :-shift] &= read[:, shift:]
        starts[:, -shift:] = False
    return starts
