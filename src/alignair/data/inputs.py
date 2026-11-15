"""Nucleotide read normalisation: the one rule a read goes through before the file reader keeps it and before a
model sees it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..core.tokens import encode
from ..errors import InputError, error_context
from .batching import positive_width, pad_vectors

BASES = frozenset("ACGTN")
# The IUPAC ambiguity codes: read as N, and not counted as unusable characters.
IUPAC_AMBIGUITY = frozenset("RYSWKMBDHV")
# The file reader drops a read more than this share of whose characters are unusable (neither a base nor an IUPAC code).
FILE_MAX_UNUSABLE = 0.2


@dataclass(frozen=True)
class SequenceNormalizationResult:
    """One value's normalised read, or the reason (``error``, ``detail``) it was refused (then ``sequence`` is None)."""
    sequence: str | None
    error: str | None = None
    detail: str | None = None
    was_cropped: bool = False


def normalize_nucleotide(value, *, max_length: int | None = None, crop: bool = False,
                         max_unusable: float | None = None) -> SequenceNormalizationResult:
    """``value`` stripped, upper-cased and every character that is not A, C, G, T or N made N. Refused when it is not
    a string (``invalid_type``), is empty after the strip (``empty``), has more than ``max_unusable`` of characters
    that are neither a base nor an IUPAC code (``ambiguous``), or is longer than ``max_length`` (``overlength``) unless
    ``crop`` cuts it to ``max_length``."""
    if not isinstance(value, str):
        return SequenceNormalizationResult(None, "invalid_type", "Sequence must be a string")
    stripped = value.strip()
    if not stripped:
        return SequenceNormalizationResult(None, "empty", "Empty sequence after strip")
    normalized = stripped.upper()
    cleaned, unusable = [], 0
    for char in normalized:
        if char in BASES:
            cleaned.append(char)
        else:
            cleaned.append("N")
            unusable += char not in IUPAC_AMBIGUITY
    if max_unusable is not None and unusable / len(normalized) > max_unusable:
        return SequenceNormalizationResult(None, "ambiguous",
                                           "Sequence exceeds the allowed fraction of unusable characters")
    sequence = "".join(cleaned)
    cropped = max_length is not None and len(sequence) > max_length
    if cropped:
        if not crop:
            return SequenceNormalizationResult(None, "overlength",
                                               f"Sequence length {len(sequence)} exceeds max_len {max_length}")
        sequence = sequence[:max_length]
    return SequenceNormalizationResult(sequence, was_cropped=cropped)


def normalize_file_read(value) -> SequenceNormalizationResult:
    """A read from a file: normalised (``normalize_nucleotide``), and dropped when more than ``FILE_MAX_UNUSABLE`` of
    it is unusable."""
    return normalize_nucleotide(value, max_unusable=FILE_MAX_UNUSABLE)


def prepare_nucleotide_inputs(sequences, max_length: int) -> tuple[list[str], list[bool]]:
    """The reads every prediction stage uses, alignment and reporting included: each normalised and cut to the
    ``max_length`` window, with a flag per read for the cut ones. A ConfigurationError for a window that is not a
    positive integer, before any read is read; an InputError for a batch that is a bare value or is empty, and for a
    read that is not a string or is empty (named ``seq_<index>``; no read after it is read)."""
    with error_context(field="max_length"):
        window = positive_width(max_length)
    reads, cropped = [], []
    for index, value in enumerate(sequence_batch_iterator(sequences)):
        with error_context(stage="sequence_input_validation", record_index=index, record_id=f"seq_{index}"):
            result = normalize_nucleotide(value, max_length=window, crop=True)
            if result.error is not None:
                raise InputError(f"seq_{index}: {result.detail}", context={"reason": result.error})
        reads.append(result.sequence)
        cropped.append(result.was_cropped)
    if not reads:
        raise InputError("cannot predict an empty nucleotide batch", context={"reason": "empty_batch"})
    return reads, cropped


def tokenize_batch(reads, width: int, device="cpu") -> dict[str, torch.Tensor]:
    """The model input of ``reads`` (``prepare_nucleotide_inputs``'s): ``tokenized_sequence``, each read's tokens
    right-padded to ``width``, on ``device``."""
    return {"tokenized_sequence": pad_vectors([encode(read) for read in reads], width=width).to(device)}


def sequence_batch_iterator(sequences):
    """An iterator over a batch's values, refusing a bare string, bytes or mapping (it would be split)."""
    if isinstance(sequences, (str, bytes, Mapping)):
        raise InputError("supply a batch of nucleotide strings, not a bare string, bytes or mapping")
    try:
        return iter(sequences)
    except TypeError as exc:
        raise InputError("sequences must be an iterable of nucleotide strings") from exc
