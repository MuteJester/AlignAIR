"""Nucleotide and protein vocabularies with explicit padding policies."""
from __future__ import annotations
import numpy as np

from ..errors import InputError
from .batching import pad_vectors

TOKEN_DICT = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}


class CenterPaddedTokenizer:
    """Encode a nucleotide string to ints and center-pad to ``max_length``.

    Vocabulary size is 6 (0..5), matching the model embedding. Unknown chars map
    to N(5). Padding token is 0; left pad = floor(pad/2), right pad = ceil.
    """

    def __init__(self, max_length: int = 576, token_dict: dict | None = None):
        self.max_length = int(max_length)
        self.token_dict = token_dict or dict(TOKEN_DICT)
        self._n = self.token_dict["N"]

    def encode(self, sequence: str) -> np.ndarray:
        return np.array([self.token_dict.get(nt, self._n) for nt in sequence], dtype=np.int64)

    def encode_and_pad(self, sequence: str) -> tuple[np.ndarray, int]:
        encoded = self.encode(sequence)
        if len(encoded) > self.max_length:
            raise ValueError(
                f"sequence length {len(encoded)} exceeds max_length {self.max_length}")
        pad_total = self.max_length - len(encoded)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = np.pad(encoded, (pad_left, pad_right), constant_values=0)
        return padded, pad_left


def pad_tokenize(sequences, token_dict: dict | None = None):
    """Right-pad a batch of nucleotide strings to the batch max length.

    Returns (tokens LongTensor (B, Lmax), mask BoolTensor (B, Lmax) with True=valid).
    Unknown characters map to N; pad token is 0.
    """
    td = token_dict or TOKEN_DICT
    n = td["N"]
    encoded = [[td.get(c, n) for c in s.upper()] for s in sequences]
    tokens = pad_vectors(encoded)
    return tokens, tokens != 0


_CANONICAL = "ACDEFGHIKLMNPQRSTVWY"          # 20 residues, alphabetical by 1-letter code
AA_TOKEN_DICT: dict[str, int] = {aa: i + 1 for i, aa in enumerate(_CANONICAL)}
AA_TOKEN_DICT["X"] = 21
AA_TOKEN_DICT["*"] = 22
AA_VOCAB_SIZE = 23                            # 0 pad + 20 residues + X + *


class InvalidResidueError(InputError):
    """Raised by the tokenizer when on_invalid='raise' and a non-vocab residue is seen."""
    code = "ALIGNAIR_AA_INVALID_RESIDUE"


class AAProteinTokenizer:
    """Encode a protein string to int ids. ``on_invalid``: 'X' maps unknown residues to X; 'raise' errors."""

    def __init__(self, token_dict: dict | None = None, on_invalid: str = "X"):
        if on_invalid not in ("X", "raise"):
            raise InputError(f"on_invalid must be 'X' or 'raise', got {on_invalid!r}")
        self.token_dict = token_dict or dict(AA_TOKEN_DICT)
        self.on_invalid = on_invalid
        self._x = self.token_dict["X"]

    def encode(self, sequence: str) -> np.ndarray:
        out = []
        for ch in sequence.upper():
            tok = self.token_dict.get(ch)
            if tok is not None:
                out.append(tok)
            elif self.on_invalid == "raise":
                raise InvalidResidueError(f"invalid residue {ch!r}", context={"residue": ch})
            else:
                out.append(self._x)
        return np.array(out, dtype=np.int64)


def pad_tokenize_aa(sequences, token_dict: dict | None = None, on_invalid: str = "X"):
    """Right-pad a batch of protein strings. Returns (LongTensor (B,Lmax), BoolTensor mask True=valid)."""
    tok = AAProteinTokenizer(token_dict, on_invalid)
    encoded = [tok.encode(s) for s in sequences]
    tokens = pad_vectors(encoded)
    return tokens, tokens != 0
