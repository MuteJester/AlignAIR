"""Configuration for the unified DNAlignAIR model."""
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DNAlignAIRConfig":
        return cls(**d)
