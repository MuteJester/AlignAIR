"""Merge per-batch raw model outputs into a single :class:`Predictions` bundle.

A boundary is the argmax of the model's masked position logits (``decode``); outputs without logits keep the
model's soft-argmax expectation. A model with anchor heads gives its V and J anchors, decoded by argmax over the read's
positions and the null slot (``decode.decoded_anchors``); productivity is derived from them later
(``predict.junction``).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ..core.anchors import ANCHOR_GENES
from .decode import decode_anchors, decode_boundaries
from .outputs import ModelOutputs
from .state import Predictions

# Predictions' marker of an anchor that is not observed.
NOT_OBSERVED = -1


def _flat(values) -> np.ndarray:
    return np.concatenate([np.asarray(value).reshape(-1) for value in values])


def _mask(mask) -> torch.Tensor:
    return torch.as_tensor(mask).bool()


def expectations(batches: Sequence[ModelOutputs], gene: str) -> tuple[np.ndarray, np.ndarray]:
    """The gene's soft-argmax (start, end) expectations over all batches."""
    return _flat(b.genes[gene].start for b in batches), _flat(b.genes[gene].end for b in batches)


def _decoded(batch: ModelOutputs, gene: str):
    """One batch's argmax (start, end) of the gene, or None without its logits or the slot masks."""
    if gene not in batch.boundary_logits or batch.start_mask is None or batch.end_mask is None:
        return None
    start_logits, end_logits = batch.boundary_logits[gene]
    return decode_boundaries(torch.as_tensor(start_logits), torch.as_tensor(end_logits), _mask(batch.start_mask),
                             _mask(batch.end_mask))


def _boundaries(batches: Sequence[ModelOutputs], gene: str):
    """The gene's reported (start, end) over all batches: the logits' argmax when every batch has them, else the
    soft expectations."""
    decoded = [_decoded(b, gene) for b in batches]
    if any(d is None for d in decoded):
        return expectations(batches, gene)
    return (np.concatenate([d[0].numpy() for d in decoded]), np.concatenate([d[1].numpy() for d in decoded]))


def _anchors(batches: Sequence[ModelOutputs]) -> dict[str, np.ndarray] | None:
    """The decoded V and J anchors over all batches (``NOT_OBSERVED`` for the null slot), or None unless every batch
    has anchor logits. An anchor is a codon start of the read's start slots."""
    if not batches or any(b.anchor_logits is None for b in batches):
        return None
    anchors = {}
    for gene in ANCHOR_GENES:
        null = np.asarray(batches[0].anchor_logits[gene]).shape[-1] - 1
        positions = np.concatenate([decode_anchors(b.anchor_logits[gene], _mask(b.start_mask)).numpy()
                                    for b in batches])
        anchors[gene] = np.where(positions == null, NOT_OBSERVED, positions)
    return anchors


def _d_presence(batches: Sequence[ModelOutputs]) -> np.ndarray | None:
    """The D presence head's class per read (the argmax of its logits), or None unless every batch has the head."""
    if not batches or any(b.d_presence_logits is None for b in batches):
        return None
    return np.vstack([np.asarray(b.d_presence_logits) for b in batches]).argmax(-1)


def clean(batches: Sequence[ModelOutputs], genes) -> Predictions:
    """The batches' outputs merged into one ``Predictions``."""
    allele, start, end = {}, {}, {}
    for g in genes:
        allele[g] = np.vstack([np.asarray(b.genes[g].allele) for b in batches])
        start[g], end[g] = _boundaries(batches, g)
    rows = len(allele[genes[0]])
    orientation = (_flat(b.orientation for b in batches).astype(int)
                   if all(b.orientation is not None for b in batches)
                   else np.zeros(rows, dtype=int))
    return Predictions(
        allele=allele, start=start, end=end,
        mutation_rate=_flat(b.mutation_rate for b in batches),
        indel_count=_flat(b.indel_count for b in batches),
        orientation=orientation, anchors=_anchors(batches), d_presence=_d_presence(batches),
        read_locus=(_flat(b.read_locus for b in batches).astype(int)
                    if batches and all(b.read_locus is not None for b in batches) else None),
    )
