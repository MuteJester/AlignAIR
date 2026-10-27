"""Sequence normalization with one source-order identity for accepted records and failures."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

from ..errors import ConfigurationError, DataContractError, InputError, error_context
from .batching import _width


@dataclass
class SequenceRecord:
    sequence_id: str
    sequence: str


@dataclass
class SequenceInputFailure:
    sequence_id: str
    error: str
    detail: str

    def to_dict(self) -> dict:
        return {"sequence_id": self.sequence_id, "error": self.error, "detail": self.detail}


@dataclass(frozen=True)
class SequenceValidationPolicy:
    alphabet: frozenset[str]
    max_length: int
    stop_symbol: str | None = None

    def __post_init__(self):
        with error_context(field="max_length"):
            object.__setattr__(self, "max_length", _width(self.max_length))
        try:
            alphabet = frozenset(self.alphabet)
        except TypeError as exc:
            raise ConfigurationError("normalization alphabet must be an iterable of characters") from exc
        if not alphabet or any(not isinstance(char, str) or len(char) != 1 or char.isspace()
                               or char != char.upper() for char in alphabet):
            raise ConfigurationError("normalization alphabet must contain single uppercase non-whitespace characters")
        if self.stop_symbol is not None and (not isinstance(self.stop_symbol, str)
                                             or len(self.stop_symbol) != 1 or self.stop_symbol in alphabet
                                             or self.stop_symbol.isspace() or self.stop_symbol != self.stop_symbol.upper()):
            raise ConfigurationError("stop_symbol must be one forbidden character or None")
        object.__setattr__(self, "alphabet", alphabet)


class _Issue(NamedTuple):
    code: str
    detail: str


def _normalize_value(value, policy):
    if not isinstance(value, str):
        return _Issue("invalid_type", "Sequence must be a string")
    stripped = value.strip()
    if not stripped:
        return _Issue("empty", "Empty sequence after strip")
    if any(char.isspace() for char in stripped):
        return _Issue("interior_whitespace", "Sequence contains interior whitespace")
    if len(stripped) > policy.max_length:
        return _Issue("overlength", f"Sequence length {len(stripped)} exceeds max_len {policy.max_length}")
    normalized = stripped.upper()
    if len(normalized) > policy.max_length:
        return _Issue("overlength", f"Sequence length {len(normalized)} exceeds max_len {policy.max_length}")
    for char in normalized:
        if char not in policy.alphabet:
            if char == policy.stop_symbol:
                return _Issue("invalid_character", f"Sequence contains stop character '{char}' which is not allowed")
            return _Issue("invalid_character",
                          f"Sequence contains invalid character '{char}'; valid: {sorted(policy.alphabet)}")
    return normalized


@dataclass(frozen=True)
class NormalizedSequenceBatch:
    """One outcome per source item; IDs never substitute for positional identity."""
    outcomes: tuple[SequenceRecord | SequenceInputFailure, ...]

    @property
    def records(self) -> tuple[SequenceRecord, ...]:
        return tuple(value for value in self.outcomes if isinstance(value, SequenceRecord))

    @property
    def errors(self) -> tuple[SequenceInputFailure, ...]:
        return tuple(value for value in self.outcomes if isinstance(value, SequenceInputFailure))

    @property
    def valid_indices(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.outcomes) if isinstance(value, SequenceRecord))

    def restore_order(self, predictions) -> list:
        """Interleave one prediction per accepted record with the original failures."""
        indices = self.valid_indices
        try:
            predictions = tuple(predictions)
        except TypeError as exc:
            raise DataContractError("predictions must be an iterable aligned with accepted records") from exc
        if len(predictions) != len(indices):
            raise DataContractError("prediction count disagrees with accepted input count",
                                    context={"expected": len(indices), "received": len(predictions)})
        ordered = list(self.outcomes)
        for index, prediction in zip(indices, predictions):
            ordered[index] = prediction
        return ordered


def normalize_sequence_inputs(sequences, policy: SequenceValidationPolicy, *, on_invalid="error") -> NormalizedSequenceBatch:
    """Normalize once, retaining every original position and deterministic seq_<index> fallback IDs.

    A bare string is one record; a bare mapping/bytes value is a format error. Within an iterable,
    mappings supply sequence/sequence_id and other items are sequence values. Error mode stops
    consumption at the first invalid record. Skip mode retains a structured failure in its slot.
    """
    if not isinstance(on_invalid, str) or on_invalid not in ("error", "skip"):
        raise ConfigurationError(f"on_invalid must be 'error' or 'skip', got {on_invalid!r}")
    if not isinstance(policy, SequenceValidationPolicy):
        raise ConfigurationError("normalization requires a SequenceValidationPolicy")
    if isinstance(sequences, (Mapping, bytes)):
        raise TypeError("sequences must be a str, an iterable of str, or an iterable of {sequence,sequence_id} dicts — not a single dict or bytes value")
    source = (sequences,) if isinstance(sequences, str) else sequences
    outcomes = []
    for index, item in enumerate(source):
        if isinstance(item, Mapping):
            provided_id, sequence = item.get("sequence_id"), item.get("sequence", "")
        else:
            provided_id, sequence = None, item
        sequence_id = provided_id if isinstance(provided_id, str) and provided_id else f"seq_{index}"
        with error_context(stage="sequence_input_validation", record_index=index, record_id=sequence_id):
            value = _normalize_value(sequence, policy)
            if isinstance(value, _Issue):
                if on_invalid == "error":
                    raise InputError(f"{sequence_id}: {value.detail}", context={"reason": value.code})
                outcomes.append(SequenceInputFailure(sequence_id, value.code, value.detail))
            else:
                outcomes.append(SequenceRecord(sequence_id, value))
    return NormalizedSequenceBatch(tuple(outcomes))
