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

# conv-tower kernel schedules (len == N+1: N convs + 1 residual).
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
    # Allele-discrimination auto-scaling: size the per-gene classification path (feature tower + the
    # prototype/latent dim) to that gene's allele count A. None -> legacy fixed dims (block_out/filters,
    # latent = A*latent_size_factor). A float (e.g. 2.0) enables scaling. Math: at convergence the A
    # class prototypes collapse to a simplex equiangular tight frame, which only exists in dimension
    # d >= A-1; so multiplier >= 1 lets the prototypes reach that maximally-separated configuration, and
    # >1 buys angular margin for SNP-similar sibling alleles.
    cls_scale_multiplier: Optional[float] = None
    cls_dim_min: int = 256         # floor: small loci keep a sane discrimination space
    cls_dim_max: int = 1024        # cap: bound the O(A^2) classification params for large unions
    cls_filter_cap: int = 256      # cap on the auto-raised classification-tower width
    state_head: bool = False       # add the per-position edit-state head (germline/sub/ins/del)
    # post-hoc allele-confidence calibration: per-gene temperature {"v":T,...} applied to the sigmoid
    # allele probabilities at inference (sigmoid(logit(p)/T)); fitted post-training, argmax-preserving.
    allele_temperatures: Optional[dict] = None

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

    def cls_spec(self, spec: "GeneSpec") -> tuple:
        """Per-gene classification-path sizing ``(cls_tower_out, cls_tower_filters, prototype_latent)``.

        Legacy (``cls_scale_multiplier is None``): the fixed ``block_out`` / ``filters`` and
        ``latent = A * latent_size_factor`` (byte-identical to pre-scaling checkpoints). Scaled: a
        discrimination width ``d = clamp(round(mult*A), cls_dim_min, cls_dim_max)`` sized so the A allele
        prototypes can reach the simplex-ETF configuration (``d >= A-1``); the tower's width is raised so
        it can actually emit ``d`` independent features (``filters * l_final >= d``), floored at the
        default ``filters`` and capped at ``cls_filter_cap``."""
        A = spec.allele_count
        if self.cls_scale_multiplier is None:
            return self.block_out, self.filters, spec.latent(self.latent_size_factor)
        d = int(max(self.cls_dim_min, min(self.cls_dim_max, round(self.cls_scale_multiplier * A))))
        l_final = max(1, self.max_seq_length >> len(spec.cls_kernels))     # tower halves L len(kernels)x
        filters = int(max(self.filters, min(self.cls_filter_cap, -(-d // l_final))))   # ceil(d / l_final)
        return d, filters, d

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
