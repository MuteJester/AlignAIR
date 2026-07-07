"""Batched model forward: raw nucleotide sequences -> list of per-batch output dicts (numpy)."""
from __future__ import annotations

import numpy as np
import torch

from ..data.tokenizer import TOKEN_DICT

_PAD = TOKEN_DICT["P"]
_N = TOKEN_DICT["N"]


def _tokenize(sequences, max_len: int) -> torch.Tensor:
    out = torch.full((len(sequences), max_len), _PAD, dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids = [TOKEN_DICT.get(c, _N) for c in str(seq).upper()][:max_len]
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


@torch.no_grad()
def run_model(model, sequences, cfg, device: str = "cpu") -> list:
    model.eval()
    raw = []
    for i in range(0, len(sequences), cfg.batch_size):
        tokens = _tokenize(sequences[i : i + cfg.batch_size], cfg.max_seq_length).to(device)
        out = model({"tokenized_sequence": tokens})
        raw.append({k: v.detach().cpu().numpy() for k, v in out.items()})
    return raw
