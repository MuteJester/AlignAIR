from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AIRRConfig:
    """AIRRistotle v2: a pure Llama decoder LM for copy-based VDJ alignment. Backbone params match
    a small Llama; `v_shortlist` controls how many V candidates the coarse filter puts in the prompt."""
    vocab_size: int
    d_model: int = 1024        # ~150M with the settings below
    n_layers: int = 14
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 2816
    max_seq: int = 8192
    rope_base: float = 10000.0
    init_std: float = 0.02        # Llama initializer_range
    v_shortlist: int = 16         # number of V candidates the coarse filter places in the prompt
