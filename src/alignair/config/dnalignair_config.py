"""Configuration for the unified DNAlignAIR model.

Single architecture path: shared nucleotide encoder backbone, RegionTagger region head,
retrieval allele caller, and the seed-and-extend banded-DP germline aligner (band head +
BandHead center + SeedExtendAligner). There are no backbone/aligner/caller choice fields —
those alternatives were removed. Only the neural-contribution ablation toggles remain.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(eq=True)
class DNAlignAIRConfig:
    d_model: int = 128
    n_layers: int = 4
    nhead: int = 8
    dim_feedforward: int = 512
    max_len: int = 1024
    orientation_dim: int = 64
    n_regions: int = 8     # len(REGIONS)
    n_states: int = 4      # germline/substitution/insertion/deletion
    band_width: int = 16   # seed_extend banded-DP half-width (the +-w window the band head places)
    # --- neural-contribution ablation toggles (spec §5.1; full-neural defaults, flipped for the
    #     Gate-3 defense that this is a learned aligner + exact structured decoder, not classical) ---
    band_features: str = "full"      # band head inputs: "full" (base-match + learned cosine [+kmer/boundary]) | "raw" (raw base-match only)
    dp_emissions: str = "learned"    # DP emission: "learned" (token reps + projections + scale) | "raw" (raw +1/-1 base-match only)
    use_learned_reps: bool = True    # False -> emission/band use the raw base-match channel only
    use_reliability: bool = True     # state-conditioned SHM reliability into the DP emission
    reader: str = "dp"               # final allele score: "dp" (log-partition, rule 1) | "pooled" | "maxsim"
    encoder_mode: str = "trained"    # ablation: "trained" | "frozen" | "random" (is the learned encoder load-bearing?)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DNAlignAIRConfig":
        return cls(**d)
