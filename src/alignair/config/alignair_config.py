"""Configuration for the faithful PyTorch AlignAIR aligner.

One model — :class:`alignair.models.AlignAIR` — serves both the old single- and multi-chain cases.
Which one you get is pure data: build the config from **one** GenAIRR dataconfig and it behaves like
the old SingleChain (no chain_type head); build it from **several** and it behaves like the old
MultiChain (union allele counts + a chain_type/locus head). Allele counts, ``has_d`` and
``num_chain_types`` are all derived for you by :meth:`AlignAIRConfig.from_dataconfigs`.

``gene_specs`` is the single source of truth for which genes exist and how each is shaped — every
consumer (model + loss) iterates it, so ``has_d`` never has to be re-expanded into a gene list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# TF-faithful conv-tower kernel schedules (len == N+1: N convs + 1 residual).
_SEG_KERNELS = (3, 3, 3, 2, 5)          # meta + all segmentation towers (N=4)
_VJ_CLS_KERNELS = (3, 3, 3, 2, 2, 2, 5)  # V/J classification towers (N=6)
_D_CLS_KERNELS = (3, 3, 2, 2, 5)         # D classification tower (shallower, N=4)


@dataclass(frozen=True)
class GeneSpec:
    """Everything that makes one gene (V/D/J) what it is, in one place: allele count, conv-tower
    kernel schedules, allele-head latent width, and whether it carries the short-D span penalty."""
    name: str                                  # "v" / "d" / "j"
    allele_count: int
    seg_kernels: tuple = _SEG_KERNELS
    cls_kernels: tuple = _VJ_CLS_KERNELS
    latent_size: Optional[int] = None
    short_d_penalty: bool = False

    def latent(self, factor: int) -> int:
        return self.latent_size if self.latent_size is not None else self.allele_count * factor


@dataclass(eq=True)
class AlignAIRConfig:
    v_allele_count: int
    j_allele_count: int
    d_allele_count: int = 0
    has_d: bool = True
    num_chain_types: int = 1       # >1 => multi-chain: adds the chain_type (locus) head + loss
    max_seq_length: int = 576
    vocab_size: int = 6            # PAD, A, C, G, T, N
    embed_dim: int = 32
    filters: int = 128             # conv feature-tower channel count
    block_out: int = 576           # ConvResidualFeatureExtractionBlock projected output dim
    latent_size_factor: int = 2    # allele-head mid width = count * factor (unless overridden)
    v_allele_latent_size: Optional[int] = None
    d_allele_latent_size: Optional[int] = None
    j_allele_latent_size: Optional[int] = None

    @property
    def gene_specs(self) -> tuple:
        """Ordered (V, [D], J) GeneSpecs — the canonical gene list the model and loss both iterate."""
        specs = [GeneSpec("v", self.v_allele_count, latent_size=self.v_allele_latent_size)]
        if self.has_d:
            specs.append(GeneSpec("d", self.d_allele_count, cls_kernels=_D_CLS_KERNELS,
                                  latent_size=self.d_allele_latent_size, short_d_penalty=True))
        specs.append(GeneSpec("j", self.j_allele_count, latent_size=self.j_allele_latent_size))
        return tuple(specs)

    def latent(self, gene: str) -> int:
        counts = {"v": self.v_allele_count, "d": self.d_allele_count, "j": self.j_allele_count}
        override = getattr(self, f"{gene}_allele_latent_size")
        return override if override is not None else counts[gene] * self.latent_size_factor

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dataconfigs(cls, *dataconfigs, max_seq_length: int = 576, **kwargs) -> "AlignAIRConfig":
        """Build straight from GenAIRR dataconfigs — derives union allele counts, ``has_d`` and the
        number of distinct chain types. One dataconfig -> single-chain; several -> multi-chain."""
        from ..reference.reference_set import ReferenceSet
        ref = ReferenceSet.from_dataconfigs(*dataconfigs)
        seen: list = []
        for dc in dataconfigs:
            ct = getattr(dc.metadata, "chain_type", None)
            key = getattr(ct, "value", str(ct))
            if key not in seen:
                seen.append(key)
        return cls.from_reference(ref, num_chain_types=max(1, len(seen)),
                                  max_seq_length=max_seq_length, **kwargs)

    @classmethod
    def from_reference(cls, reference_set, *, num_chain_types: int = 1,
                       max_seq_length: int = 576, **kwargs) -> "AlignAIRConfig":
        """Build from an already-constructed :class:`ReferenceSet` (allele counts come from it)."""
        return cls(
            v_allele_count=len(reference_set.gene("V")),
            j_allele_count=len(reference_set.gene("J")),
            d_allele_count=len(reference_set.gene("D")) if reference_set.has_d else 0,
            has_d=reference_set.has_d,
            num_chain_types=num_chain_types,
            max_seq_length=max_seq_length,
            **kwargs,
        )
