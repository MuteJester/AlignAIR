"""Merge per-batch raw model outputs into a single :class:`Predictions` bundle.

A boundary is the argmax of the model's masked position logits (``decode``); outputs without logits keep the
model's soft-argmax expectation. A model with anchor heads gives its V and J anchors, decoded by argmax over the read's
positions and the null slot (``decode.decoded_anchors``); productivity is derived from them later
(``predict.junction``).
"""
from __future__ import annotations

import numpy as np

from ..core.anchors import ANCHOR_GENES, ANCHOR_LOGITS
from ..core.d_presence import D_PRESENCE_LOGITS
from .decode import decoded_anchors, decoded_boundaries
from .state import Predictions

# Predictions' marker of an anchor that is not observed.
NOT_OBSERVED = -1


def _flat(batches, key):
    return np.concatenate([np.asarray(b[key]).reshape(-1) for b in batches])


def _boundaries(batches, gene):
    """The gene's reported (start, end) over all batches: the logits' argmax, else the soft expectations."""
    decoded = [decoded_boundaries(b, gene) for b in batches]
    if any(d is None for d in decoded):
        return _flat(batches, f"{gene}_start"), _flat(batches, f"{gene}_end")
    return (np.concatenate([d[0].numpy() for d in decoded]), np.concatenate([d[1].numpy() for d in decoded]))


def _anchors(batches) -> dict[str, np.ndarray] | None:
    """The decoded V and J anchors over all batches (``NOT_OBSERVED`` for the null slot), or None without anchor
    logits."""
    decoded = [decoded_anchors(b) for b in batches]
    if not decoded or any(d is None for d in decoded):
        return None
    anchors = {}
    for gene in ANCHOR_GENES:
        null = np.asarray(batches[0][ANCHOR_LOGITS[gene]]).shape[-1] - 1
        positions = np.concatenate([d[gene].numpy() for d in decoded])
        anchors[gene] = np.where(positions == null, NOT_OBSERVED, positions)
    return anchors


def _d_presence(batches) -> np.ndarray | None:
    """The D presence head's class per read (the argmax of its logits), or None without the head."""
    if not batches or not all(D_PRESENCE_LOGITS in b for b in batches):
        return None
    return np.vstack([np.asarray(b[D_PRESENCE_LOGITS]) for b in batches]).argmax(-1)


def clean(raw_batches: list, genes) -> Predictions:
    allele, start, end = {}, {}, {}
    for g in genes:
        allele[g] = np.vstack([np.asarray(b[f"{g}_allele"]) for b in raw_batches])
        start[g], end[g] = _boundaries(raw_batches, g)
    rows = len(allele[genes[0]])
    orientation = (_flat(raw_batches, "orientation").astype(int)
                   if all("orientation" in b for b in raw_batches)
                   else np.zeros(rows, dtype=int))
    return Predictions(
        allele=allele, start=start, end=end,
        mutation_rate=_flat(raw_batches, "mutation_rate"),
        indel_count=_flat(raw_batches, "indel_count"),
        orientation=orientation, anchors=_anchors(raw_batches), d_presence=_d_presence(raw_batches),
    )
