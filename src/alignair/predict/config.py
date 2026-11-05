"""Configuration for the prediction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.boundaries import read_window


@dataclass
class PredictConfig:
    max_seq_length: int = 576
    has_d: bool = True
    selector: str = "absolute"        # derived, calibration-free set rule (p >= threshold)
    threshold: float = 0.5            # the set's cut on the allele scores (absolute); a percentage for the relative rule
    cap: int = 3                      # max alleles per gene
    germline_reader: str = "heuristic"  # germline aligner: "heuristic" (anchored + Cython DP) or "wfa"
    batch_size: int = 64
    airr: bool = True                 # assemble full AIRR fields (junction/CDR/FWR/np/identity); off = light records
    pad_mode: str = "right"           # reads are right-padded
    genotype: Optional[dict] = None   # {gene: set(allele names)} enables genotype constraint (subset)
    genotype_method: str = "mask"  # mask | renormalize (Bayes posterior) | redistribute (legacy)
    chain_types: Optional[tuple] = None  # ordered locus names of the reference; one locus labels every record

    @property
    def window(self) -> int:
        """The longest read the model sees: the padded width, less the reserved end slot."""
        return read_window(self.max_seq_length)
