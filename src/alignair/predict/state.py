"""Immutable typed state that flows through the prediction pipeline.

Each stage consumes one of these and returns the next — no shared mutable blob. Per-gene data is
keyed by ``'v'``/``'d'``/``'j'`` (D absent for light chains).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

GENES = ("v", "d", "j")


@dataclass(frozen=True)
class GeneCall:
    """A per-read allele call for one gene: selected allele names (top-1 first) + likelihoods."""
    names: tuple[str, ...]
    likelihoods: tuple[float, ...]


@dataclass
class Predictions:
    """Cleaned, batch-merged model outputs (numpy). Per-gene dicts keyed by present genes."""
    allele: dict[str, np.ndarray]     # {gene: [N, n_alleles] sigmoid probabilities}
    start: dict[str, np.ndarray]      # {gene: [N] float read position}
    end: dict[str, np.ndarray]        # {gene: [N] float read position}
    mutation_rate: np.ndarray         # [N]
    indel_count: np.ndarray           # [N]
    productive: np.ndarray            # [N] bool
    orientation: Optional[np.ndarray] = None   # [N] predicted orientation id (model self-corrects)


@dataclass
class Segments:
    """De-padded, clipped, order-repaired integer segment coordinates in the read frame."""
    start: dict[str, np.ndarray]      # {gene: [N] int}
    end: dict[str, np.ndarray]        # {gene: [N] int}


@dataclass(frozen=True)
class GermlineAlignment:
    """Germline alignment of one read's top-1 allele for one gene."""
    allele: str
    seq_start: int
    seq_end: int
    germ_start: int
    germ_end: int
    cigar: str
