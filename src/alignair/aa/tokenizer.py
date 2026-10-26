"""Amino-acid tokenizer. PAD is id 0 with NO character ('P' is proline). Right-padding.

Stop ``*`` and unknown ``X`` are distinct tokens. Stop-rejection for input reads is an
input-contract concern enforced at the pipeline layer (Phase 2), not here.
"""
from __future__ import annotations

from alignair.errors import InputError

import numpy as np
import torch

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
    lmax = max((len(e) for e in encoded), default=0)
    tokens = torch.zeros(len(encoded), lmax, dtype=torch.long)
    mask = torch.zeros(len(encoded), lmax, dtype=torch.bool)
    for i, e in enumerate(encoded):
        if len(e):
            tokens[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            mask[i, :len(e)] = True
    return tokens, mask
