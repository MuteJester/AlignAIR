"""Neural building blocks used by the AlignAIR model:
  heads/    — region + state label constants and the in-model orientation head
  weighting — Kendall uncertainty loss weighting
"""
from .heads import (
    RegionTagger, decode_boundaries, REGIONS, REGION_INDEX,
    PerPositionStateHead, state_counts, state_reliability, STATES, STATE_INDEX,
    OrientationHead, apply_orientation, NUM_ORIENTATIONS,
)

__all__ = [
    "RegionTagger", "decode_boundaries", "REGIONS", "REGION_INDEX",
    "PerPositionStateHead", "state_counts", "state_reliability", "STATES", "STATE_INDEX",
    "OrientationHead", "apply_orientation", "NUM_ORIENTATIONS",
]
