"""Protein-product normalization policy and historical record/result imports."""
from ...data.inputs import (NormalizedSequenceBatch, SequenceInputFailure as ErrorResult,
                           SequenceRecord as Record, SequenceValidationPolicy, normalize_sequence_inputs)
from ...data.tokenizer import AA_TOKEN_DICT

VALID_AMINO_ACIDS = frozenset(AA_TOKEN_DICT) - {'*'}


def normalize_input_batch(sequences, on_invalid: str = "error", max_len: int = 192) -> NormalizedSequenceBatch:
    """Apply the product alphabet while preserving source positions for predictions and failures."""
    policy = SequenceValidationPolicy(VALID_AMINO_ACIDS, max_len, stop_symbol='*')
    return normalize_sequence_inputs(sequences, policy, on_invalid=on_invalid)


def normalize_inputs(sequences, on_invalid: str = "error", max_len: int = 192) -> tuple[list[Record], list[ErrorResult]]:
    """Historical split-list interface, delegated to the indexed normalization owner."""
    batch = normalize_input_batch(sequences, on_invalid=on_invalid, max_len=max_len)
    return list(batch.records), list(batch.errors)
