"""The nucleotide token vocabulary, the label no loss reads and the default read window: one definition each, for
every part of AlignAIR that encodes a read or masks a label."""
from __future__ import annotations

import numpy as np

# P (0) pads a read to its batch's width; the bases A, T, G and C are 1 to 4, in the order of ``BASES``; N (5) is any
# other character.
TOKEN_DICT = {"P": 0, "A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
PAD, N = TOKEN_DICT["P"], TOKEN_DICT["N"]
BASES = "ATGC"                    # BASES[k] is the base of token k + 1
VOCAB_SIZE = len(TOKEN_DICT)
# A token's complement, by token id: A and T swap, G and C swap; N and the pad stay.
COMPLEMENT_TOKENS = (PAD, TOKEN_DICT["T"], TOKEN_DICT["A"], TOKEN_DICT["C"], TOKEN_DICT["G"], N)
IGNORE_INDEX = -100               # a label no loss reads (torch's cross-entropy default)
DEFAULT_WINDOW = 576              # a model's padded read width, its ``max_seq_length``

_BASE_TOKEN = {base: TOKEN_DICT[base] for base in BASES}


def encode(sequence: str) -> np.ndarray:
    """``sequence``'s token ids (int64), in either case; a character that is not A, C, G or T is N."""
    return np.array([_BASE_TOKEN.get(char, N) for char in sequence.upper()], dtype=np.int64)


def is_base(tokens):
    """Where ``tokens`` (an array or a tensor) holds a base (A, C, G or T): not the pad and not N."""
    return (tokens >= 1) & (tokens <= len(BASES))
