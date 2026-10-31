"""Configuration for the faithful PyTorch AlignAIR aligner.

One model — :class:`alignair.core.AlignAIR` — serves nucleotide and protein-class configurations.
Which one you get is pure data: build the config from **one** GenAIRR dataconfig and it behaves like
the old SingleChain (no chain_type head); build it from **several** and it behaves like the old
MultiChain (union allele counts + a chain_type/locus head). Allele counts, ``has_d`` and
``num_chain_types`` are all derived for you by :meth:`AlignAIRConfig.from_dataconfigs`.

``gene_specs`` is the single source of truth for which genes exist and how each is shaped — every
consumer (model + loss) iterates it, so ``has_d`` never has to be re-expanded into a gene list.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

from ..errors import ConfigurationError

# conv-tower kernel schedules (len == N+1: N convs + 1 residual).
_SEG_KERNELS = (3, 3, 3, 2, 5)          # meta + all segmentation towers (N=4)
_VJ_CLS_KERNELS = (3, 3, 3, 2, 2, 2, 5)  # V/J classification towers (N=6)
_D_CLS_KERNELS = (3, 3, 2, 2, 5)         # D classification tower (shallower, N=4)


def _reference_short_d_penalty(reference, *, has_d, classification_mode):
    """Only a final Short-D label supports this legacy multilabel objective.

    Keep the legacy value for configurations where the objective is inactive, preserving
    continuation of historical non-D and categorical checkpoints.
    """
    return (not has_d or classification_mode != "multilabel"
            or bool(reference.gene("D").names) and reference.gene("D").names[-1] == "Short-D")


def _reference_productive_unknown(*, use_meta_heads):
    """Reference-bound models leave unknown productivity out of the productive objective.

    Keep the historical value where the objective is inactive (no metadata heads), preserving
    continuation of historical checkpoints without those heads.
    """
    return "masked" if use_meta_heads else "negative"


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
    # latent dim) to that gene's allele count A. None -> legacy fixed dims (block_out/filters,
    # latent = A*latent_size_factor). A float (e.g. 2.0) enables a bounded width heuristic.
    # Width alone does not guarantee feature rank, class separation or improved accuracy.
    cls_scale_multiplier: Optional[float] = None
    cls_dim_min: int = 256         # floor: small loci keep a sane discrimination space
    cls_dim_max: int = 1024        # cap: bound the O(A^2) classification params for large unions
    cls_filter_cap: int = 256      # cap on the auto-raised classification-tower width
    state_head: bool = False       # add the per-position edit-state head (germline/sub/ins/del)
    # in-model orientation (reverse-complement detect/correct/re-embed). Nucleotide default. Set
    # False for alphabets with no reverse complement (amino acids) — the RC path is both undefined
    # and its complement table is nucleotide-only. Kept True by default so nt checkpoints are
    # byte-identical.
    use_orientation: bool = True
    # post-hoc allele-confidence calibration: per-gene temperature {"v":T,...} applied to the sigmoid
    # allele probabilities at inference (sigmoid(logit(p)/T)); fitted post-training. Clipping and
    # finite precision can merge scores into ties; downstream set rules retain their own semantics.
    allele_temperatures: Optional[dict] = None
    # Serialized construction choices shared by nucleotide and protein models. Architecture names
    # resolve through core.components; loading never imports arbitrary code from a model file.
    architecture: str = "conv-v1"
    classification_mode: Literal["multilabel", "categorical"] = "multilabel"
    boundary_mode: Literal["residue", "half_open"] = "residue"
    use_meta_heads: bool = True
    reject_empty_sequences: bool = False
    input_space: Literal["nucleotide", "protein"] = "nucleotide"
    label_space: Literal["alleles", "protein_classes"] = "alleles"
    # Historical count-only/CSV configurations assign the last D column to Short-D. Reference-
    # bound constructors derive this setting from label identity; old serialized configs retain True.
    short_d_penalty: bool = True
    # How the productive objective treats a record whose productivity is unknown (None): "masked"
    # leaves it out of the productive loss and metric; "negative" is the historical objective, which
    # trained it as non-productive. Reference-bound constructors select "masked" when the metadata
    # heads are on; serialized configs without the field keep "negative".
    productive_unknown: Literal["negative", "masked"] = "negative"
    # The heads read from a trunk's per-position features: only an architecture with a trunk builds them
    # (``core.model.TRUNK_HEAD_FLAGS``). New ones are appended here, so positional construction never shifts.
    # The per-position recombination-region head (pre/V/N1/D/N2/J/post).
    region_head: bool = False
    # The V and J anchor heads (the V Cys and J Trp/Phe codon starts, or "not observed"). The junction, frame, stop
    # codon and productivity are then derived from them (``predict.junction``), and the learned productive head and
    # its loss term are left out.
    anchor_heads: bool = False
    # The 3-state D presence head (absent / too short to call / callable, ``core.d_presence``), which gates the D call
    # at prediction. Only a heavy-chain configuration (``has_d``) has a D to gate.
    d_presence_head: bool = False
    # The chip's training inputs (``data.chip_inputs``): per-base germline columns and identifiable regions, and per
    # batch the bank masks and teacher inputs. The chip reads half-open boundaries.
    chip_inputs: bool = False

    def __post_init__(self):
        for name in ("short_d_penalty", "region_head", "anchor_heads", "d_presence_head", "chip_inputs"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be a boolean", context={"field": name})
        if self.d_presence_head and not self.has_d:
            raise ConfigurationError("d_presence_head needs a D: the configuration has none (has_d is False)",
                                     context={"field": "d_presence_head"},
                                     hint="Leave d_presence_head off for a light chain.")
        if self.chip_inputs and self.boundary_mode != "half_open":
            raise ConfigurationError("chip_inputs needs half-open boundaries", context={"field": "chip_inputs"},
                                     hint="Set boundary_mode='half_open'.")
        for name, choices in (("classification_mode", ("multilabel", "categorical")),
                              ("boundary_mode", ("residue", "half_open")),
                              ("input_space", ("nucleotide", "protein")),
                              ("label_space", ("alleles", "protein_classes")),
                              ("productive_unknown", ("negative", "masked"))):
            if getattr(self, name) not in choices:
                raise ConfigurationError(f"unsupported {name}: {getattr(self, name)!r}",
                                         context={"field": name, "allowed": choices})

    def with_amino_acid_heads(self) -> "AlignAIRConfig":
        """Copy this config with the established protein-class model's head and mask choices.

        The caller supplies its tokenizer vocabulary and label counts. Other combinations, such as
        categorical heads with auxiliary predictions, can be selected with dataclasses.replace.
        """
        return replace(self, classification_mode="categorical", boundary_mode="half_open",
                       use_orientation=False, use_meta_heads=False, state_head=False,
                       reject_empty_sequences=True, input_space="protein", label_space="protein_classes")

    def classification_key(self, gene: str) -> str:
        """Output key with explicit score semantics: sigmoid allele probabilities or class logits."""
        suffix = "allele" if self.classification_mode == "multilabel" else "class_logits"
        return f"{gene}_{suffix}"

    @property
    def gene_specs(self) -> tuple:
        """Ordered (V, [D], J) GeneSpecs — the canonical gene list the model and loss both iterate."""
        specs = [GeneSpec("v", self.v_allele_count, latent_size=self.v_allele_latent_size)]
        if self.has_d:
            specs.append(GeneSpec("d", self.d_allele_count, cls_kernels=_D_CLS_KERNELS,
                                  latent_size=self.d_allele_latent_size, short_d_penalty=self.short_d_penalty))
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
        width ``d = clamp(round(mult*A), cls_dim_min, cls_dim_max)`` sets both projected and latent
        dimensions. Tower channels are raised toward ``ceil(d / l_final)``, floored at the default
        ``filters`` and capped at ``cls_filter_cap``. These bounds can limit available dimensions;
        the setting budgets parameters without guaranteeing independent features or class separation."""
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
        kwargs.setdefault("short_d_penalty", _reference_short_d_penalty(reference_set,
            has_d=reference_set.has_d, classification_mode=kwargs.get("classification_mode", "multilabel")))
        kwargs.setdefault("productive_unknown",
                          _reference_productive_unknown(use_meta_heads=kwargs.get("use_meta_heads", True)))
        return cls(
            v_allele_count=len(reference_set.gene("V")),
            j_allele_count=len(reference_set.gene("J")),
            d_allele_count=len(reference_set.gene("D")) if reference_set.has_d else 0,
            has_d=reference_set.has_d,
            num_chain_types=num_chain_types,
            max_seq_length=max_seq_length,
            **kwargs,
        )
