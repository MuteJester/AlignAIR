"""Tokenizer-bound model inputs with explicit sequence-window and empty-input policies."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..errors import ConfigurationError, DataContractError, InputError, error_context
from .batching import _width, pad_vectors


@dataclass(frozen=True)
class SequenceInputAdapter:
    tokenizer: object
    max_length: int
    device: str = "cpu"
    overflow: str = "error"
    allow_empty: bool = False
    alphabet: frozenset[str] | None = None
    vocab_size: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "max_length", _width(self.max_length))
        if self.vocab_size is not None:
            object.__setattr__(self, "vocab_size", _width(self.vocab_size))
        if not callable(getattr(self.tokenizer, "encode", None)):
            raise ConfigurationError("inference tokenizers must supply encode(sequence)")
        try:
            object.__setattr__(self, "device", torch.device(self.device))
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ConfigurationError("invalid inference device", context={"field": "device"}) from exc
        if self.overflow not in ("error", "prefix") or type(self.allow_empty) is not bool:
            raise ConfigurationError("choose an error/prefix window and a boolean empty-input policy")
        if self.alphabet is not None:
            try:
                alphabet = frozenset(self.alphabet)
            except TypeError as exc:
                raise ConfigurationError("alphabet must be an iterable of characters") from exc
            if not alphabet or any(not isinstance(value, str) or len(value) != 1 for value in alphabet):
                raise ConfigurationError("alphabet must contain individual allowed characters")
            object.__setattr__(self, "alphabet", alphabet)

    def validate(self, sequences) -> tuple[str, ...]:
        if isinstance(sequences, (str, bytes)):
            raise InputError("supply a batch of sequences, not a bare string")
        try:
            sequences = tuple(sequences)
        except TypeError as exc:
            raise InputError("sequences must be an iterable of strings") from exc
        for row, sequence in enumerate(sequences):
            with error_context(record_index=row):
                if not isinstance(sequence, str) or not sequence and not self.allow_empty:
                    raise InputError("inference sequences must be nonempty strings")
                if self.overflow == "error" and len(sequence) > self.max_length:
                    raise InputError("sequence exceeds the inference window", context={"max_length": self.max_length})
                if self.alphabet is not None and not set(sequence) <= self.alphabet:
                    raise InputError("sequence contains characters outside the declared alphabet")
        return sequences

    def __call__(self, sequences) -> dict[str, torch.Tensor]:
        rows = []
        for row, sequence in enumerate(self.validate(sequences)):
            with error_context(stage="inference_tokenization", record_index=row):
                sequence = sequence[:self.max_length] if self.overflow == "prefix" else sequence
                encoded = self.tokenizer.encode(sequence)
                try:
                    tokens = torch.as_tensor(encoded)
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise DataContractError("tokenizer output must be an integer vector") from exc
                if (tokens.ndim != 1 or tokens.device.type != "cpu" or tokens.dtype == torch.bool
                        or tokens.is_floating_point() or tokens.is_complex() or tokens.numel() != len(sequence)):
                    raise DataContractError("tokenization must preserve one integer CPU token per sequence position")
                if (tokens < 0).any() or self.vocab_size is not None and (tokens >= self.vocab_size).any():
                    raise DataContractError("token IDs fall outside the declared vocabulary")
                if not self.allow_empty and not (tokens != 0).any():
                    raise DataContractError("tokenizer output contains no observed tokens")
                rows.append(tokens)
        if not rows:
            raise InputError("cannot adapt an empty inference batch")
        return {"tokenized_sequence": pad_vectors(rows, width=self.max_length).to(self.device)}
