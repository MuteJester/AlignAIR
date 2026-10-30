"""The conserved V and J anchors a model with anchor heads locates (``AlignAIRConfig.anchor_heads``): the one
definition of their codon length, their genes and their keys.

An anchor is the read position of the first base of a conserved codon: V's Cys, J's Trp/Phe. The junction runs from
the V anchor to the end of the J anchor's codon (``predict.junction``). A model scores each read position as an anchor,
plus a null slot ("not observed"); its targets (``data``), loss terms and weights (``tasks``) and records (``predict``)
are keyed by ``ANCHOR_KEYS``, its outputs by ``ANCHOR_LOGITS``.
"""
from __future__ import annotations

CODON = 3                                              # bases in an anchor codon
ANCHOR_GENES = ("v", "j")                              # the genes with a conserved anchor, in loss-weight order
ANCHOR_KEYS = {gene: f"{gene}_anchor" for gene in ANCHOR_GENES}            # targets, loss terms, record fields
ANCHOR_LOGITS = {gene: f"{gene}_anchor_logits" for gene in ANCHOR_GENES}   # model outputs (B, L + 1)
