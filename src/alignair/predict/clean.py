"""Merge per-batch raw model outputs into a single :class:`Predictions` bundle.

A boundary is the argmax of the model's masked position logits (``decode``); outputs without logits keep the
model's soft-argmax expectation. ``productive`` is thresholded at 0.5.
"""
from __future__ import annotations

import numpy as np

from .decode import decoded_boundaries
from .state import Predictions


def _stack(batches, key):
    return np.concatenate([np.asarray(b[key]).reshape(len(np.asarray(b[key])), -1) for b in batches])


def _flat(batches, key):
    return np.concatenate([np.asarray(b[key]).reshape(-1) for b in batches])


def _boundaries(batches, gene):
    """The gene's reported (start, end) over all batches: the logits' argmax, else the soft expectations."""
    decoded = [decoded_boundaries(b, gene) for b in batches]
    if any(d is None for d in decoded):
        return _flat(batches, f"{gene}_start"), _flat(batches, f"{gene}_end")
    return (np.concatenate([d[0].numpy() for d in decoded]), np.concatenate([d[1].numpy() for d in decoded]))


def clean(raw_batches: list, genes) -> Predictions:
    allele, start, end = {}, {}, {}
    for g in genes:
        allele[g] = np.vstack([np.asarray(b[f"{g}_allele"]) for b in raw_batches])
        start[g], end[g] = _boundaries(raw_batches, g)
    productive = _flat(raw_batches, "productive") > 0.5
    orientation = (_flat(raw_batches, "orientation").astype(int)
                   if all("orientation" in b for b in raw_batches)
                   else np.zeros(len(productive), dtype=int))
    chain_type = (np.vstack([np.asarray(b["chain_type_logits"]) for b in raw_batches]).argmax(-1)
                  if all("chain_type_logits" in b for b in raw_batches) else None)
    state_logits = (np.concatenate([np.asarray(b["state_logits"]) for b in raw_batches], axis=0)
                    if raw_batches and all("state_logits" in b for b in raw_batches) else None)
    return Predictions(
        allele=allele, start=start, end=end,
        mutation_rate=_flat(raw_batches, "mutation_rate"),
        indel_count=_flat(raw_batches, "indel_count"),
        productive=productive, orientation=orientation, chain_type=chain_type,
        state_logits=state_logits,
    )
