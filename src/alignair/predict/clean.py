"""Merge per-batch raw model outputs into a single :class:`Predictions` bundle.

Our model emits soft-argmax coordinate *expectations* (scalars), so positions are taken directly
(no argmax-over-logits re-discretization). ``productive`` is thresholded at 0.5.
"""
from __future__ import annotations

import numpy as np

from .state import Predictions


def _stack(batches, key):
    return np.concatenate([np.asarray(b[key]).reshape(len(np.asarray(b[key])), -1) for b in batches])


def _flat(batches, key):
    return np.concatenate([np.asarray(b[key]).reshape(-1) for b in batches])


def clean(raw_batches: list, genes) -> Predictions:
    allele, start, end = {}, {}, {}
    for g in genes:
        allele[g] = np.vstack([np.asarray(b[f"{g}_allele"]) for b in raw_batches])
        start[g] = _flat(raw_batches, f"{g}_start")
        end[g] = _flat(raw_batches, f"{g}_end")
    productive = _flat(raw_batches, "productive") > 0.5
    orientation = (_flat(raw_batches, "orientation").astype(int)
                   if all("orientation" in b for b in raw_batches)
                   else np.zeros(len(productive), dtype=int))
    return Predictions(
        allele=allele, start=start, end=end,
        mutation_rate=_flat(raw_batches, "mutation_rate"),
        indel_count=_flat(raw_batches, "indel_count"),
        productive=productive, orientation=orientation,
    )
