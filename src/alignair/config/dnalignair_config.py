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
    backbone: str = "conv"  # "conv" (conv-stem + Transformer) | "shared" (RoPE/SDPA/SwiGLU SharedNucleotideEncoder)
    aligner: str = "softdp"  # germline aligner: "softdp" (gap-aware DP) | "diagonal" (legacy cosine corr) | "pointer" (fast parallel) | "seed_extend" (shared encoder + banded exact DP)
    band_half_width: int = 0  # pointer aligner indel band half-width (0 = single diagonal)
    region_decoder: str = "linear"  # region head: "linear" (RegionTagger) | "query" (mask-span decoder w/ boundary posteriors)
    caller: str = "retrieval"  # allele caller: "retrieval" (cosine vs germline embeddings) | "classifier" (masked per-allele head; NOT dynamic-genotype-compliant, excluded from seed_extend)
    allele_counts: dict | None = None  # {"V":198,"D":33,"J":7} required when caller=="classifier"
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
