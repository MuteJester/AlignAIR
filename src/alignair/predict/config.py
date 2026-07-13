"""Configuration for the prediction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictConfig:
    max_seq_length: int = 576
    has_d: bool = True
    selector: str = "absolute"        # derived, calibration-free set rule (p >= threshold)
    threshold: float = 0.5            # BCE-calibrated decision boundary (absolute); or pct for legacy
    cap: int = 3                      # max alleles per gene
    germline_reader: str = "heuristic"  # germline aligner: "heuristic" (anchored + Cython DP) or "wfa"
    batch_size: int = 64
    airr: bool = True                 # assemble full AIRR fields (junction/CDR/FWR/np/identity); off = light records
    pad_mode: str = "right"           # our trainer right-pads (TF used "center")
    genotype: Optional[dict] = None   # {gene: set(allele names)} enables genotype constraint (subset)
    genotype_method: str = "mask"  # mask | renormalize (Bayes posterior) | redistribute (legacy)
    chain_types: Optional[tuple] = None  # ordered locus names; maps multi-chain chain_type index -> locus
    allele_temperatures: Optional[dict] = None  # per-gene temperature for allele-confidence calibration
