"""Neural building blocks for DNAlignAIR, grouped by component:
  primitives/ — generic layers (activations, conv, embedding, masking)
  encoder/    — sequence encoders (backbone, shared nucleotide encoder, germline encoder)
  heads/      — task heads (region, state, orientation, allele matching, segmentation)
  aligner/    — germline alignment (base-match, diagonal ops, soft-DP, pointer, band head, banded DP)
  weighting   — Kendall uncertainty loss weighting
"""
from .encoder import SequenceBackbone, SharedNucleotideEncoder, GermlineEncoder
from .heads import (
    RegionTagger, decode_boundaries, REGIONS, REGION_INDEX,
    PerPositionStateHead, state_counts, state_reliability, STATES, STATE_INDEX,
    OrientationHead, apply_orientation, NUM_ORIENTATIONS,
    AlleleMatchingHead, multilabel_match_loss,
)
from .aligner import GermlineAligner, decode_germline_coords

__all__ = [
    "SequenceBackbone", "SharedNucleotideEncoder", "GermlineEncoder",
    "RegionTagger", "decode_boundaries", "REGIONS", "REGION_INDEX",
    "PerPositionStateHead", "state_counts", "state_reliability", "STATES", "STATE_INDEX",
    "OrientationHead", "apply_orientation", "NUM_ORIENTATIONS",
    "AlleleMatchingHead", "multilabel_match_loss",
    "GermlineAligner", "decode_germline_coords",
]
