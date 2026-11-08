"""The typed state that flows through the prediction pipeline.

``PredictionState`` is what the stages (``predict.stages``) pass along: each returns a copy with its own fields set.
The values it holds are the stages' types below. Per-gene data is keyed by ``'v'``/``'d'``/``'j'`` (D absent for light
chains).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PredictConfig

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
    model: object
    reference: object
    cfg: PredictConfig
    device: str


@dataclass(frozen=True)
class PredictionState:
    """One prediction's state, stage by stage: the input, then the field each stage sets (None until it has run)."""
    context: PredictionContext
    sequences: object                                   # the input as given (any iterable of strings)
    reads: Optional[list] = None                        # input_policy: post-crop, pre-orientation
    cropped: Optional[list] = None                      # input_policy: whether each read was cropped
    batches: Optional[list] = None                      # forward: the network's outputs, one dict per batch
    predictions: Optional[Predictions] = None           # clean
    canonical: Optional[list] = None                    # canonicalize: the reads in the model's forward frame
    segments: Optional[Segments] = None                 # segments
    calls: Optional[dict] = None                        # alleles: gene -> one GeneCall per read
    alignments: Optional[dict] = None                   # germline, then better_v: gene -> GermlineAlignment | None
    no_v: Optional[list] = None                         # v_presence: whether each read has no V
    v_ties: Optional[list] = None                       # ties: each read's V call and the alleles tied with it
    records: Optional[list] = None                      # records: one record dict per read
    rows: Optional[list] = None                         # airr: what predict returns

    @classmethod
    def start(cls, model, reference, cfg: PredictConfig, device: str, sequences) -> "PredictionState":
        """The state before any stage: the context and the input."""
        return cls(PredictionContext(model, reference, cfg, device), sequences)

