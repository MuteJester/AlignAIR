"""Center-padded nucleotide tokenizer (port of CenterPaddedSequenceTokenizer)."""
import numpy as np

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
