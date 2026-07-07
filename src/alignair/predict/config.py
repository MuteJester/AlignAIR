"""Configuration for the prediction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictConfig:
    max_seq_length: int = 576
    has_d: bool = True
    threshold_pct: float = 0.1        # MaxLikelihoodPercentageThreshold: keep p_i >= pct*max(p)
    cap: int = 3                      # max alleles per gene
    batch_size: int = 64
    pad_mode: str = "right"           # our trainer right-pads (TF used "center")
    genotype: Optional[dict] = None   # {gene: set(allele names)} enables genotype likelihood correction
