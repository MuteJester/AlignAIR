"""Configuration for the faithful PyTorch AlignAIR aligner (single- and multi-chain).

1:1 port of the TF ``SingleChainAlignAIR``/``MultiChainAlignAIR`` hyperparameters. Allele counts
and ``has_d`` come from the dataconfig / ``ReferenceSet``; latent sizes default to ``count * 2``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(eq=True)
class AlignAIRConfig:
    v_allele_count: int
    j_allele_count: int
    d_allele_count: int = 0
    has_d: bool = True
    max_seq_length: int = 576
    vocab_size: int = 6            # PAD, A, C, G, T, N
    embed_dim: int = 32
    filters: int = 128             # conv feature-tower channel count
    block_out: int = 576           # ConvResidualFeatureExtractionBlock projected output dim
    latent_size_factor: int = 2    # allele-head mid width = count * factor (unless overridden)
    v_allele_latent_size: Optional[int] = None
    d_allele_latent_size: Optional[int] = None
    j_allele_latent_size: Optional[int] = None

    def latent(self, gene: str) -> int:
        count = {"v": self.v_allele_count, "d": self.d_allele_count, "j": self.j_allele_count}[gene]
        override = getattr(self, f"{gene}_allele_latent_size")
        return override if override is not None else count * self.latent_size_factor
