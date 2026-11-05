"""The nucleotide vocabulary."""
from __future__ import annotations
import numpy as np

TOKEN_DICT = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}


class CenterPaddedTokenizer:
    """Encode a nucleotide string to token ids (``TOKEN_DICT``: vocabulary 0..5, 0 the padding token, unknown
    characters to N). Batches are right-padded by ``data.inference.SequenceInputAdapter``. (The class name is
    historical; the reads are not center-padded.)"""

    def __init__(self, max_length: int = 576, token_dict: dict | None = None):
        self.max_length = int(max_length)
        self.token_dict = token_dict or dict(TOKEN_DICT)
        self._n = self.token_dict["N"]

    def encode(self, sequence: str) -> np.ndarray:
        return np.array([self.token_dict.get(nt, self._n) for nt in sequence], dtype=np.int64)
