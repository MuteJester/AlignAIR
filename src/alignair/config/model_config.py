"""Typed model configuration for AlignAIR models.

Decouples the model from the raw GenAIRR ``DataConfig`` and makes models
deterministically reconstructable from a saved JSON config.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass(eq=True)
class ModelConfig:
    max_seq_length: int
    v_allele_count: int
    j_allele_count: int
    d_allele_count: Optional[int]
    has_d_gene: bool

    # Optional explicit latent sizes; when None, derived as count * latent_size_factor.
    v_allele_latent_size: Optional[int] = None
    d_allele_latent_size: Optional[int] = None
    j_allele_latent_size: Optional[int] = None
    latent_size_factor: int = 2

    # Activations.
    classification_mid_activation: str = "swish"
    feature_block_activation: str = "tanh"

    # Multi-chain (None / empty for single-chain).
    number_of_chains: Optional[int] = None
    chain_types: Optional[List[str]] = field(default=None)

    def __post_init__(self) -> None:
        if self.has_d_gene and self.d_allele_count is None:
            raise ValueError("has_d_gene=True requires a non-None d_allele_count")
        if not self.has_d_gene and self.d_allele_count is not None:
            # Tolerate but normalize: a non-D model carries no D count.
            self.d_allele_count = None

    @property
    def v_latent_dim(self) -> int:
        return self.v_allele_latent_size or self.v_allele_count * self.latent_size_factor

    @property
    def j_latent_dim(self) -> int:
        return self.j_allele_latent_size or self.j_allele_count * self.latent_size_factor

    @property
    def d_latent_dim(self) -> Optional[int]:
        if not self.has_d_gene:
            return None
        return self.d_allele_latent_size or self.d_allele_count * self.latent_size_factor

    @property
    def is_multi_chain(self) -> bool:
        return bool(self.number_of_chains and self.number_of_chains > 1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)

    @classmethod
    def from_dataconfig(cls, dataconfig, **overrides) -> "ModelConfig":
        """Build from a single GenAIRR ``DataConfig``."""
        has_d = bool(dataconfig.metadata.has_d)
        return cls(
            max_seq_length=overrides.pop("max_seq_length"),
            v_allele_count=dataconfig.number_of_v_alleles,
            j_allele_count=dataconfig.number_of_j_alleles,
            d_allele_count=dataconfig.number_of_d_alleles if has_d else None,
            has_d_gene=has_d,
            **overrides,
        )

    @classmethod
    def from_dataconfigs(cls, container, **overrides) -> "ModelConfig":
        """Build from a multi-chain ``MultiDataConfigContainer``."""
        has_d = bool(container.has_at_least_one_d())
        try:
            chain_types = [getattr(ct, "value", str(ct)) for ct in container.chain_types()]
        except Exception:
            chain_types = None
        return cls(
            max_seq_length=overrides.pop("max_seq_length"),
            v_allele_count=container.number_of_v_alleles,
            j_allele_count=container.number_of_j_alleles,
            d_allele_count=container.number_of_d_alleles if has_d else None,
            has_d_gene=has_d,
            number_of_chains=len(chain_types) if chain_types else None,
            chain_types=chain_types,
            **overrides,
        )
