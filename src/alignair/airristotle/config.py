from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AIRRConfig:
    vocab_size: int
    d_model: int = 1024        # ~150M with the settings below
    n_layers: int = 14
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 2816
    max_seq: int = 8192
    rope_base: float = 10000.0
