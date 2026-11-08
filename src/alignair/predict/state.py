"""The typed state that flows through the prediction pipeline.

``PredictionState`` is what the stages (``predict.stages``) pass along: each returns a copy with its own fields set.
The values it holds are the stages' types below. Per-gene data is keyed by ``'v'``/``'d'``/``'j'`` (D absent for light
chains).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import numpy as np

from .config import PredictConfig

if TYPE_CHECKING:                       # annotations only: results imports this module, and torch is heavy
    import torch

    from ..reference.reference_set import ReferenceSet
    from .outputs import ModelOutputs
    from .results import ReadResult

GENES = ("v", "d", "j")


@dataclass(frozen=True)
class AlignResult:
    """One segment's alignment to its germline window: the CIGAR (M/I/D runs only) and the read and germline
    coordinates it spans (0-based, end-exclusive)."""
    score: float        # higher is better
    cigar: str
    q_start: int        # first read base consumed
    q_end: int          # one past the last read base
    t_start: int        # germline bases before the alignment
    t_end: int          # germline end (exclusive)


@dataclass(frozen=True)
class GeneCall:
    """A per-read allele call for one gene: selected allele names (top-1 first) + likelihoods."""
    names: tuple[str, ...]
    likelihoods: tuple[float, ...]


@dataclass
class Predictions:
    """Cleaned, batch-merged model outputs (numpy). Per-gene dicts keyed by present genes."""
    allele: dict[str, np.ndarray]     # {gene: [N, n_alleles] scores relative to the read's best allele (best = 1)}
    start: dict[str, np.ndarray]      # {gene: [N] float read position}
    end: dict[str, np.ndarray]        # {gene: [N] float read position}
    mutation_rate: np.ndarray         # [N]
    indel_count: np.ndarray           # [N]
    orientation: Optional[np.ndarray] = None   # [N] predicted orientation id (model self-corrects)
    # {"v"/"j": [N] int} the decoded anchors (anchor heads only): the forward-frame read position of the V Cys / J
    # Trp-Phe codon's first base, -1 when the anchor is not observed
    anchors: Optional[dict[str, np.ndarray]] = None
    # [N] int: the D presence head's class (core.d_presence: absent, short, callable), which gates the D call
    d_presence: Optional[np.ndarray] = None

    def take(self, rows) -> "Predictions":
        """The predictions of ``rows`` (read indices), in that order."""
        return _taken(self, rows)


@dataclass
class Segments:
    """De-padded, clipped, order-repaired integer segment coordinates in the read frame.

    All coordinates satisfy ``0 <= start <= end <= seq_len`` and the biological ordering
    ``v_end <= d_start <= d_end <= j_start <= j_end`` jointly (constrained projection, not a
    one-directional repair). A segment squeezed to zero length is *absent* rather than a manufactured
    one-base span. ``low_quality[i]`` flags reads whose mandatory V anchor collapsed to zero length —
    no feasible segmentation exists for that read (e.g. read shorter than the called layout)."""
    start: dict[str, np.ndarray]      # {gene: [N] int}
    end: dict[str, np.ndarray]        # {gene: [N] int}
    low_quality: Optional[np.ndarray] = None   # [N] bool: V anchor collapsed -> infeasible segmentation

    def take(self, rows) -> "Segments":
        """The segments of ``rows`` (read indices), in that order."""
        return _taken(self, rows)


@dataclass(frozen=True)
class GermlineAlignment:
    """Germline alignment of one read's top-1 allele for one gene."""
    allele: str
    seq_start: int
    seq_end: int
    germ_start: int
    germ_end: int
    cigar: str


@dataclass(frozen=True)
class PredictionContext:
    """What a prediction runs with: the model, its bound reference, the configuration and the device."""
    model: torch.nn.Module
    reference: ReferenceSet
    cfg: PredictConfig
    device: str


@dataclass(frozen=True)
class PredictionState:
    """One prediction's state, stage by stage: the input, then the field each stage sets (None until it has run)."""
    context: PredictionContext
    sequences: Optional[Iterable[str]]                  # the input as given (any iterable of strings)
    # input: an orientation id per read to teacher-force (None: the model's own)
    orientation_given: Optional[tuple[Optional[int], ...]] = None
    reads: Optional[list[str]] = None                   # input_policy: post-crop, pre-orientation
    cropped: Optional[list[bool]] = None                # input_policy: whether each read was cropped
    batches: Optional[list[ModelOutputs]] = None        # forward: the network's outputs, one per batch
    predictions: Optional[Predictions] = None           # clean
    canonical: Optional[list[str]] = None               # canonicalize: the reads in the model's forward frame
    segments: Optional[Segments] = None                 # segments
    calls: Optional[dict[str, list[GeneCall]]] = None   # alleles: gene -> one call per read
    # germline, then better_v: gene -> one alignment per read (None: no call)
    alignments: Optional[dict[str, list[Optional[GermlineAlignment]]]] = None
    no_v: Optional[list[bool]] = None                   # v_presence: whether each read has no V
    v_ties: Optional[list[tuple[str, ...]]] = None      # ties: each read's V call and the alleles tied with it
    results: Optional[list[ReadResult]] = None          # records: one typed result per read
    rows: Optional[list[dict]] = None                   # airr: what predict returns

    @classmethod
    def start(cls, model, reference, cfg: PredictConfig, device: str, sequences, *,
              orientation=None) -> "PredictionState":
        """The state before any stage: the context, the input, and the orientation ids to teacher-force (one per read,
        None for the model's own; None: none forced)."""
        return cls(PredictionContext(model, reference, cfg, device), sequences,
                   None if orientation is None else tuple(orientation))

    def take(self, rows) -> "PredictionState":
        """The state of ``rows`` (read indices), in that order: every per-read field (``PER_READ_FIELDS``) cut to them,
        the context kept, and the input and the batches, which are not per read, dropped."""
        return replace(self, sequences=None, batches=None,
                       **{name: _pick(getattr(self, name), list(rows)) for name in PER_READ_FIELDS})


# The PredictionState fields that hold one entry per read (as a list, an array, a gene dict of them, or Predictions or
# Segments): what ``PredictionState.take`` cuts.
PER_READ_FIELDS = ("orientation_given", "reads", "cropped", "predictions", "canonical", "segments", "calls",
                   "alignments", "no_v", "v_ties", "results", "rows")


def _pick(value, rows: list):
    """The entries of ``rows`` of a per-read value."""
    if value is None:
        return None
    if isinstance(value, (Predictions, Segments)):
        return value.take(rows)
    if isinstance(value, dict):
        return {key: _pick(entry, rows) for key, entry in value.items()}
    if isinstance(value, np.ndarray):
        return value[rows]
    picked = [value[i] for i in rows]
    return tuple(picked) if isinstance(value, tuple) else picked


def _taken(value, rows):
    """A Predictions or Segments with every per-read field cut to ``rows``."""
    from dataclasses import fields
    return replace(value, **{f.name: _pick(getattr(value, f.name), list(rows)) for f in fields(value)})

