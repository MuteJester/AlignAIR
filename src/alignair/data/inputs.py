"""Sequence normalization with one source-order identity for accepted records and failures."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

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
class SequenceNormalizationResult:
    """Content normalization without record identity; rejected values have no sequence."""
    sequence: str | None
    error: str | None = None
    detail: str | None = None
    was_cropped: bool = False


@dataclass(frozen=True)
class SequenceValidationPolicy:
    alphabet: frozenset[str]
    max_length: int | None
    stop_symbol: str | None = None
    unknown_symbol: str | None = None
    ambiguity_symbols: frozenset[str] = frozenset()
    max_invalid_fraction: float | None = None
    overflow: str = "error"
    reject_interior_whitespace: bool = True

    def __post_init__(self):
        if self.max_length is not None:
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
        if self.unknown_symbol is not None and (not isinstance(self.unknown_symbol, str)
                                                or self.unknown_symbol not in alphabet):
            raise ConfigurationError("unknown_symbol must be an allowed alphabet character or None")
        try:
            ambiguity = frozenset(self.ambiguity_symbols)
        except TypeError as exc:
            raise ConfigurationError("ambiguity_symbols must be an iterable of characters") from exc
        if any(not isinstance(char, str) or len(char) != 1 or char.isspace()
               or char != char.upper() or char in alphabet for char in ambiguity):
            raise ConfigurationError("ambiguity_symbols must contain uppercase characters outside the alphabet")
        if self.unknown_symbol is None and (ambiguity or self.max_invalid_fraction is not None):
            raise ConfigurationError("ambiguity and invalid-fraction policies require an unknown_symbol")
        fraction = self.max_invalid_fraction
        if fraction is not None and (isinstance(fraction, bool) or not isinstance(fraction, Real)
                                     or not 0 <= fraction <= 1 or not math.isfinite(fraction)):
            raise ConfigurationError("max_invalid_fraction must be a finite fraction between 0 and 1")
        if (not isinstance(self.overflow, str) or self.overflow not in ("error", "prefix")
                or self.overflow == "prefix" and self.max_length is None):
            raise ConfigurationError("choose error overflow or prefix cropping with a finite window")
        if not isinstance(self.reject_interior_whitespace, bool):
            raise ConfigurationError("reject_interior_whitespace must be a boolean")
        object.__setattr__(self, "ambiguity_symbols", ambiguity)

    def normalize(self, value) -> SequenceNormalizationResult:
        """Normalize one value with the same content rules used by indexed batches."""
        return _normalize_value(value, self)


def _overlength(length, maximum):
    return SequenceNormalizationResult(None, "overlength", f"Sequence length {length} exceeds max_len {maximum}")


def _normalize_value(value, policy):
    if not isinstance(value, str):
        return SequenceNormalizationResult(None, "invalid_type", "Sequence must be a string")
    stripped = value.strip()
    if not stripped:
        return SequenceNormalizationResult(None, "empty", "Empty sequence after strip")
    if policy.reject_interior_whitespace and any(char.isspace() for char in stripped):
        return SequenceNormalizationResult(None, "interior_whitespace", "Sequence contains interior whitespace")
    # Strict alphabets retain their historical length-before-alphabet rejection priority.
    strict_window = policy.unknown_symbol is None and policy.max_length is not None and policy.overflow == "error"
    if strict_window and len(stripped) > policy.max_length:
        return _overlength(len(stripped), policy.max_length)
    normalized = stripped.upper()
    if strict_window and len(normalized) > policy.max_length:
        return _overlength(len(normalized), policy.max_length)
    cleaned, invalid = [], 0
    for char in normalized:
        if char in policy.alphabet:
            cleaned.append(char)
        elif char == policy.stop_symbol:
            return SequenceNormalizationResult(None, "invalid_character", f"Sequence contains stop character '{char}' which is not allowed")
        elif policy.unknown_symbol is None:
            return SequenceNormalizationResult(None, "invalid_character",
                                               f"Sequence contains invalid character '{char}'; valid: {sorted(policy.alphabet)}")
        else:
            cleaned.append(policy.unknown_symbol)
            invalid += char not in policy.ambiguity_symbols
    if policy.max_invalid_fraction is not None and invalid / len(normalized) > policy.max_invalid_fraction:
        return SequenceNormalizationResult(None, "ambiguous", "Sequence exceeds the allowed fraction of unusable characters")
    sequence = "".join(cleaned)
    cropped = policy.max_length is not None and len(sequence) > policy.max_length
    if cropped:
        if policy.overflow == "error":
            return _overlength(len(sequence), policy.max_length)
        sequence = sequence[:policy.max_length]
    return SequenceNormalizationResult(sequence, was_cropped=cropped)


@dataclass(frozen=True)
class NormalizedSequenceBatch:
    """One outcome per source item; IDs never substitute for positional identity."""
    outcomes: tuple[SequenceRecord | SequenceInputFailure, ...]
    cropped_indices: tuple[int, ...] = ()

    @property
    def records(self) -> tuple[SequenceRecord, ...]:
        return tuple(value for value in self.outcomes if isinstance(value, SequenceRecord))

    @property
    def errors(self) -> tuple[SequenceInputFailure, ...]:
        return tuple(value for value in self.outcomes if isinstance(value, SequenceInputFailure))

    @property
    def valid_indices(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.outcomes) if isinstance(value, SequenceRecord))

    @property
    def crop_flags(self) -> tuple[bool, ...]:
        """Crop flags aligned with accepted records; cropped_indices retains source positions."""
        cropped = set(self.cropped_indices)
        return tuple(index in cropped for index in self.valid_indices)

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


def normalize_sequence_inputs(sequences, policy: SequenceValidationPolicy, *, on_invalid="error",
                              allow_records: bool = True) -> NormalizedSequenceBatch:
    """Normalize once, retaining every original position and deterministic seq_<index> fallback IDs.

    A bare string is one record; a bare mapping/bytes value is a format error. Within an iterable,
    mappings supply sequence/sequence_id and other items are sequence values. Error mode stops
    consumption at the first invalid record. Skip mode retains a structured failure in its slot.
    """
    if not isinstance(on_invalid, str) or on_invalid not in ("error", "skip"):
        raise ConfigurationError(f"on_invalid must be 'error' or 'skip', got {on_invalid!r}")
    if not isinstance(policy, SequenceValidationPolicy):
        raise ConfigurationError("normalization requires a SequenceValidationPolicy")
    if not isinstance(allow_records, bool):
        raise ConfigurationError("allow_records must be a boolean")
    if isinstance(sequences, (Mapping, bytes)):
        raise TypeError("sequences must be a str, an iterable of str, or an iterable of {sequence,sequence_id} dicts — not a single dict or bytes value")
    source = (sequences,) if isinstance(sequences, str) else sequences
    outcomes, cropped = [], []
    for index, item in enumerate(source):
        if allow_records and isinstance(item, Mapping):
            provided_id, sequence = item.get("sequence_id"), item.get("sequence", "")
        else:
            provided_id, sequence = None, item
        sequence_id = provided_id if isinstance(provided_id, str) and provided_id else f"seq_{index}"
        with error_context(stage="sequence_input_validation", record_index=index, record_id=sequence_id):
            value = policy.normalize(sequence)
            if value.error is not None:
                if on_invalid == "error":
                    raise InputError(f"{sequence_id}: {value.detail}", context={"reason": value.error})
                outcomes.append(SequenceInputFailure(sequence_id, value.error, value.detail))
            else:
                outcomes.append(SequenceRecord(sequence_id, value.sequence))
                if value.was_cropped:
                    cropped.append(index)
    return NormalizedSequenceBatch(tuple(outcomes), tuple(cropped))


def nucleotide_file_policy(max_length: int | None = None) -> SequenceValidationPolicy:
    """Clean IUPAC/unknown bases to N, rejecting more than 20% non-IUPAC content before length."""
    return SequenceValidationPolicy(frozenset("ACGTN"), max_length, unknown_symbol="N",
                                    ambiguity_symbols=frozenset("RYSWKMBDHV"), max_invalid_fraction=0.2,
                                    reject_interior_whitespace=False)


def nucleotide_prediction_policy(max_length: int) -> SequenceValidationPolicy:
    """Normalize to ACGTN and crop once; in-memory prediction has no file-quality rejection gate."""
    with error_context(field="max_length"):
        maximum = _width(max_length)
    return SequenceValidationPolicy(frozenset("ACGTN"), maximum, unknown_symbol="N", overflow="prefix",
                                    reject_interior_whitespace=False)


def validate_nucleotide_sequence(sequence, max_len: int | None = None) -> tuple[str | None, str | None]:
    """Historical file-validator tuple shape; overlength retains its legacy too_long reason code."""
    result = nucleotide_file_policy(max_len).normalize(sequence)
    reason = "too_long" if result.error == "overlength" else result.error
    return result.sequence, reason


def validate_nucleotide(sequence) -> str | None:
    """Return a cleaned nucleotide sequence or None under the file content policy."""
    return validate_nucleotide_sequence(sequence)[0]


def prepare_nucleotide_inputs(sequences, max_length: int) -> tuple[list[str], list[bool]]:
    """Prepare a batch of strings for all prediction stages, including alignment and reporting."""
    policy = nucleotide_prediction_policy(max_length)
    batch = normalize_sequence_inputs(sequence_batch_iterator(sequences), policy, allow_records=False)
    if not batch.records:
        raise InputError("cannot predict an empty nucleotide batch", context={"reason": "empty_batch"})
    return [record.sequence for record in batch.records], list(batch.crop_flags)


def sequence_batch_iterator(sequences):
    """Obtain unvalidated batch values without splitting bare strings, bytes or mapping keys."""
    if isinstance(sequences, (str, bytes, Mapping)):
        raise InputError("supply a batch of nucleotide strings, not a bare string, bytes or mapping")
    try:
        return iter(sequences)
    except TypeError as exc:
        raise InputError("sequences must be an iterable of nucleotide strings") from exc
