"""Neural building blocks for DNAlignAIR, grouped by component:
  primitives/ — generic layers (activations, conv, embedding, masking)
  encoder/    — the shared nucleotide encoder (one encoder for reads and references)
  heads/      — task heads (region, state, orientation, allele matching, segmentation)
  aligner/    — germline alignment (base-match, diagonal ops, soft-DP, band head, banded DP)
  weighting   — Kendall uncertainty loss weighting
"""
from .encoder import SharedNucleotideEncoder
from .heads import (
    RegionTagger, decode_boundaries, REGIONS, REGION_INDEX,
    PerPositionStateHead, state_counts, state_reliability, STATES, STATE_INDEX,
    OrientationHead, apply_orientation, NUM_ORIENTATIONS,
    AlleleMatchingHead, multilabel_match_loss,
)
from .aligner import decode_germline_coords

__all__ = [
    "SharedNucleotideEncoder",
    "RegionTagger", "decode_boundaries", "REGIONS", "REGION_INDEX",
    "PerPositionStateHead", "state_counts", "state_reliability", "STATES", "STATE_INDEX",
    "OrientationHead", "apply_orientation", "NUM_ORIENTATIONS",
    "AlleleMatchingHead", "multilabel_match_loss",
    "decode_germline_coords",
]
