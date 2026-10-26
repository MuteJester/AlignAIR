"""Input validation and normalization for AA inference product."""

from dataclasses import dataclass
from typing import Union, Iterable, Sequence

from ...errors import ConfigurationError, InputError


# Canonical amino acids (20) + X (unknown)
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")


@dataclass
class Record:
    """A validated and normalized amino acid sequence record."""
    sequence_id: str
    sequence: str


@dataclass
class ErrorResult:
    """Result for an invalid record."""
    sequence_id: str
    error: str  # Short error code: "empty", "invalid_character", "interior_whitespace", "overlength"
    detail: str  # Human-readable error message

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "error": self.error,
            "detail": self.detail,
        }


def normalize_inputs(
    sequences: Union[str, Iterable[str], Iterable[dict]],
    on_invalid: str = "error",
    max_len: int = 192,
) -> tuple[list[Record], list[ErrorResult]]:
    """
    Normalize and validate amino acid sequences.

    Args:
        sequences: A single sequence string, an iterable of sequence strings,
                   or an iterable of dicts with "sequence_id" (optional) and "sequence" keys.
        on_invalid: How to handle invalid records:
                    - "error" (default): raise ValueError on first invalid record
                    - "skip": return ErrorResult for invalid records, Records for valid ones
        max_len: Maximum allowed sequence length (default 192). Sequences longer than this are rejected.

    Returns:
        A tuple (records, errors) where:
        - records: list of valid Record objects
        - errors: list of ErrorResult objects (empty if on_invalid="error")

    Raises:
        ValueError: If on_invalid="error" and an invalid record is encountered.
        ValueError: If on_invalid is not "error" or "skip".
    """
    # Validate on_invalid parameter
    if on_invalid not in ("error", "skip"):
        raise ConfigurationError(f"on_invalid must be 'error' or 'skip', got {on_invalid!r}")

    records: list[Record] = []
    errors: list[ErrorResult] = []

    # Normalize input to a list of dicts
    input_records = _normalize_input_format(sequences)

    # Process each record
    for i, record_data in enumerate(input_records):
        # Use provided sequence_id only if it's a non-empty string; otherwise fallback
        provided_id = record_data.get("sequence_id")
        sequence_id = provided_id if isinstance(provided_id, str) and provided_id else f"seq_{i}"
        sequence = record_data.get("sequence", "")

        # Validate and normalize
        validation_result = _validate_sequence(
            sequence, sequence_id, max_len
        )

        if validation_result is None:
            # Valid sequence
            normalized_sequence = _normalize_sequence(sequence)
            records.append(Record(sequence_id=sequence_id, sequence=normalized_sequence))
        else:
            # Invalid sequence
            error_code, error_detail = validation_result
            error = ErrorResult(sequence_id=sequence_id, error=error_code, detail=error_detail)

            if on_invalid == "error":
                raise InputError(f"{sequence_id}: {error_detail}",
                                 context={"stage": "aa_input_validation", "record_id": sequence_id,
                                          "reason": error_code})
            else:  # on_invalid == "skip"
                errors.append(error)

    return records, errors


def _normalize_input_format(
    sequences: Union[str, Iterable[str], Iterable[dict]],
) -> list[dict]:
    """
    Convert input sequences to a list of dicts with "sequence_id" and "sequence" keys.

    A bare string is treated as a single sequence, not a file path.
    Supports mixed lists of strings and dicts.

    Raises:
        TypeError: If sequences is a bare dict (not wrapped in a list).
    """
    if isinstance(sequences, str):
        # Bare string is a single sequence
        return [{"sequence": sequences}]

    # Reject bare dict (must be wrapped in a list)
    if isinstance(sequences, dict):
        raise TypeError(
            "sequences must be a str, an iterable of str, or an iterable of {sequence,sequence_id} dicts — not a single dict"
        )

    # Must be iterable; convert to list
    seq_list = list(sequences)
    if not seq_list:
        return []

    # Handle mixed lists (strings and dicts) or homogeneous lists
    result = []
    for item in seq_list:
        if isinstance(item, dict):
            result.append(item)
        else:
            # Treat as string
            result.append({"sequence": item})
    return result


def _normalize_sequence(sequence: str) -> str:
    """Normalize a valid sequence: strip and uppercase."""
    return sequence.strip().upper()


def _validate_sequence(
    sequence: str,
    sequence_id: str,
    max_len: int,
) -> Union[tuple[str, str], None]:
    """
    Validate a sequence.

    Returns:
        None if valid, or (error_code, error_detail) if invalid.
    """
    # Strip surrounding whitespace
    stripped = sequence.strip()

    # Check if empty after stripping
    if not stripped:
        return ("empty", "Empty sequence after strip")

    # Check for interior whitespace (after strip)
    if any(c.isspace() for c in stripped):
        return ("interior_whitespace", "Sequence contains interior whitespace")

    # Check overlength
    if len(stripped) > max_len:
        return ("overlength", f"Sequence length {len(stripped)} exceeds max_len {max_len}")

    # Check alphabet (uppercase version of stripped)
    uppercase = stripped.upper()
    for char in uppercase:
        if char not in VALID_AMINO_ACIDS:
            if char == "*":
                return (
                    "invalid_character",
                    f"Sequence contains stop character '*' which is not allowed",
                )
            else:
                return (
                    "invalid_character",
                    f"Sequence contains invalid character '{char}'; valid: {sorted(VALID_AMINO_ACIDS)}",
                )

    # All checks passed
    return None
