"""The configuration of an AlignAIR model: the chip on nucleotide reads. Its alleles are not part of it: they are the
bound reference's (``AlignAIR.bind_reference``). Build one for a reference with ``core.setup.prepare_model``.
``genes`` is the one ordered gene list (V, [D], J) every consumer iterates.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index

from ..errors import ConfigurationError
from ..reference.genes import model_genes
from .tokens import DEFAULT_WINDOW, VOCAB_SIZE


@dataclass(eq=True)
class AlignAIRConfig:
    has_d: bool = True
    num_chain_types: int = 1       # the loci of the reference's locus schema
    max_seq_length: int = DEFAULT_WINDOW
    vocab_size: int = VOCAB_SIZE   # the tokens of core.tokens.TOKEN_DICT
    embed_dim: int = 32
    # The serialized network name; it resolves through core.components, and loading never imports code from a file.
    architecture: str = "chip-v1"
    # At inference, orient each read by the bound reference (``network.orientation.BankOrientation``): the transform
    # under which most of its 12-mers are germline 12-mers; the orientation head decides when none wins.
    bank_orientation: bool = True
    # The chip's trunk width (channels per position): its parser, anchors, SHM reader and D units read features this
    # wide. chip-v1 is trained at 64.
    trunk_channels: int = 64

    def __post_init__(self):
        for name in ("has_d", "bank_orientation"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be a boolean", context={"field": name})
        for name in ("num_chain_types", "max_seq_length", "embed_dim", "trunk_channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not hasattr(type(value), "__index__") or index(value) < 1:
                raise ConfigurationError(f"{name} must be a positive integer, got {value!r}", context={"field": name})
            setattr(self, name, index(value))
        if self.vocab_size != VOCAB_SIZE:
            raise ConfigurationError("model vocabulary disagrees with the nucleotide tokenizer",
                                     context={"field": "vocab_size", "expected": VOCAB_SIZE})
        if not isinstance(self.architecture, str) or not self.architecture:
            raise ConfigurationError("architecture must be a registered network name",
                                     context={"field": "architecture"})

    @property
    def genes(self) -> tuple[str, ...]:
        """The model's genes in order: ("v", "d", "j"), or ("v", "j") without a D."""
        return model_genes(self.has_d)
