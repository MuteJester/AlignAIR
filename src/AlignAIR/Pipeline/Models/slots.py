"""Typed pipeline data slots — frozen dataclasses representing stage outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from AlignAIR.Pipeline.Models.config import PipelineConfig


@dataclass(frozen=True)
class FileInfo:
    """Metadata about the input file."""
    file_path: str
    file_name: str
    file_type: str          # 'csv', 'tsv', 'fasta'
    n_sequences: int
    file_size_bytes: int


@dataclass(frozen=True)
class LoadedModel:
    """Everything loaded from the model bundle."""
    inference_wrapper: object      # SavedModelInferenceWrapper
    dataconfig: object             # MultiDataConfigContainer
    max_seq_length: int
    has_d_gene: bool
    v_allele_count: int
    j_allele_count: int
    d_allele_count: Optional[int]
    orientation_pipeline: Optional[object]
    bundle_fingerprint: str
    chain_type: str                # 'heavy', 'light', 'multi'
    # KmerDensityExtractor for candidate sequence extraction
    candidate_extractor: Optional[object] = None


class _NumpySlotMixin:
    """Mixin to prevent frozen dataclass issues with numpy array equality."""
    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        for f in self.__dataclass_fields__:
            a, b = getattr(self, f), getattr(other, f)
            if isinstance(a, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
        return True

    def __hash__(self):
        return id(self)


@dataclass(frozen=True)
class RawPredictions(_NumpySlotMixin):
    """Direct model output arrays, one per output head.
    Shape: (n_sequences, ...) for each array.
    """
    # Allele probabilities: (n_seq, n_alleles) after sigmoid
    v_allele: np.ndarray
    j_allele: np.ndarray
    d_allele: Optional[np.ndarray]  # None for light chain

    # Position logits: (n_seq, max_seq_length) raw logits
    v_start_logits: np.ndarray
    v_end_logits: np.ndarray
    j_start_logits: np.ndarray
    j_end_logits: np.ndarray
    d_start_logits: Optional[np.ndarray]
    d_end_logits: Optional[np.ndarray]

    # Scalar outputs: (n_seq, 1) or (n_seq,)
    mutation_rate: np.ndarray
    indel_count: np.ndarray
    productive: np.ndarray

    # Tokenization metadata needed by downstream stages
    padding_tokens: np.ndarray     # (n_seq,) center padding counts


@dataclass(frozen=True)
class ProcessedPredictions(_NumpySlotMixin):
    """Cleaned predictions after logit extraction and batch merging.
    Position logits replaced by argmax positions.
    """
    # Allele probabilities (same as raw, or genotype-adjusted)
    v_allele: np.ndarray
    j_allele: np.ndarray
    d_allele: Optional[np.ndarray]

    # Extracted positions: (n_seq,) integer indices
    v_start: np.ndarray
    v_end: np.ndarray
    j_start: np.ndarray
    j_end: np.ndarray
    d_start: Optional[np.ndarray]
    d_end: Optional[np.ndarray]

    # Scalars
    mutation_rate: np.ndarray
    indel_count: np.ndarray
    productive: np.ndarray

    # Padding info (needed for segment correction)
    padding_tokens: np.ndarray


@dataclass(frozen=True)
class CorrectedSegments(_NumpySlotMixin):
    """Segment positions after center-padding removal and boundary clamping."""
    v_start: np.ndarray
    v_end: np.ndarray
    j_start: np.ndarray
    j_end: np.ndarray
    d_start: Optional[np.ndarray]
    d_end: Optional[np.ndarray]


@dataclass(frozen=True)
class AlleleCall:
    """Allele call for a single gene type for a single sequence."""
    allele_names: Tuple[str, ...]
    likelihoods: Tuple[float, ...]


@dataclass(frozen=True)
class SelectedAlleles:
    """Per-sequence allele calls after thresholding."""
    v_calls: List[AlleleCall]
    j_calls: List[AlleleCall]
    d_calls: Optional[List[AlleleCall]]  # None for light chain


@dataclass(frozen=True)
class GermlineAlignment:
    """Germline alignment result for a single sequence."""
    v_sequence_alignment: str
    v_germline_alignment: str
    j_sequence_alignment: str
    j_germline_alignment: str
    d_sequence_alignment: Optional[str]
    d_germline_alignment: Optional[str]
    v_cigar: str
    j_cigar: str
    d_cigar: Optional[str]
    junction: str
    junction_aa: str


@dataclass(frozen=True)
class GermlineAlignments:
    """All germline alignments for a pipeline run."""
    alignments: List[GermlineAlignment]   # length n_sequences


@dataclass(frozen=True)
class PipelineResult:
    """Final assembled pipeline output — what gets serialized to CSV/AIRR."""
    sequences: List[str]
    file_info: FileInfo
    config: PipelineConfig
    corrected_segments: CorrectedSegments
    selected_alleles: SelectedAlleles
    germline_alignments: GermlineAlignments
    mutation_rate: np.ndarray
    indel_count: np.ndarray
    productive: np.ndarray
    chain_type: str
    # Status per sequence: 'OK', 'SKIPPED_INVALID', 'WARNING'
    status: List[str] = field(default_factory=list)
