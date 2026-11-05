"""The configuration of an AlignAIR model: the chip on nucleotide reads.

Build it from **one** GenAIRR dataconfig for one locus, or from **several** for a multi-locus model (union allele
counts); allele counts, ``has_d`` and ``num_chain_types`` are derived by :meth:`AlignAIRConfig.from_dataconfigs`.
``genes`` is the one ordered gene list (V, [D], J) every consumer iterates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ..errors import ConfigurationError


@dataclass(eq=True)
class AlignAIRConfig:
    v_allele_count: int
    j_allele_count: int
    d_allele_count: int = 0
    has_d: bool = True
    num_chain_types: int = 1       # the loci of the reference's locus schema
    max_seq_length: int = 576
    vocab_size: int = 6            # PAD, A, C, G, T, N
    embed_dim: int = 32
    # in-model orientation (reverse-complement detect/correct/re-embed)
    use_orientation: bool = True
    # The serialized network name; it resolves through core.components, and loading never imports code from a file.
    architecture: str = "chip-v1"
    reject_empty_sequences: bool = False
    input_space: Literal["nucleotide"] = "nucleotide"
    label_space: Literal["alleles"] = "alleles"
    # At inference, orient each read by the bound reference (``network.orientation.BankOrientation``): the transform
    # under which most of its 12-mers are germline 12-mers; the orientation head decides when none wins.
    bank_orientation: bool = True
    # The V and J anchor heads (the V Cys and J Trp/Phe codon starts, or "not observed"). The junction, frame, stop
    # codon and productivity are derived from them (``predict.junction``).
    anchor_heads: bool = True
    # The 3-state D presence head (absent / too short to call / callable, ``core.d_presence``), which gates the D call
    # at prediction. Only a configuration with a D has one to gate; None (the default) builds it exactly then.
    d_presence_head: Optional[bool] = None
    # The chip's trunk width (channels per position): its parser, anchors, SHM reader and D units read features this
    # wide. chip-v1 is trained at 64.
    trunk_channels: int = 64

    def __post_init__(self):
        if self.d_presence_head is None:
            self.d_presence_head = self.has_d
        for name in ("anchor_heads", "d_presence_head", "bank_orientation"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be a boolean", context={"field": name})
        if type(self.trunk_channels) is not int or self.trunk_channels < 1:
            raise ConfigurationError(f"trunk_channels must be a positive integer, got {self.trunk_channels!r}",
                                     context={"field": "trunk_channels"})
        if self.d_presence_head and not self.has_d:
            raise ConfigurationError("d_presence_head needs a D: the configuration has none (has_d is False)",
                                     context={"field": "d_presence_head"},
                                     hint="Leave d_presence_head off for a light chain.")
        if self.bank_orientation and not self.use_orientation:
            raise ConfigurationError("bank_orientation orients reads in place of the orientation head: it needs one",
                                     context={"field": "bank_orientation"}, hint="Set use_orientation=True.")
        for name, choices in (("input_space", ("nucleotide",)), ("label_space", ("alleles",))):
            if getattr(self, name) not in choices:
                raise ConfigurationError(f"unsupported {name}: {getattr(self, name)!r}",
                                         context={"field": name, "allowed": choices})

    @property
    def genes(self) -> tuple[str, ...]:
        """The model's genes in order: ("v", "d", "j"), or ("v", "j") without a D."""
        return ("v", "d", "j") if self.has_d else ("v", "j")

    def allele_count(self, gene: str) -> int:
        """The width of ``gene``'s allele output (``gene`` one of ``genes``)."""
        return getattr(self, f"{gene}_allele_count")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dataconfigs(cls, *dataconfigs, max_seq_length: int = 576, **kwargs) -> "AlignAIRConfig":
        """Build straight from GenAIRR dataconfigs — derives union allele counts, ``has_d`` and the number of
        distinct chain types."""
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
