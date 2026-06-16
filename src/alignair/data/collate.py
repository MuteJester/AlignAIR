"""Collate per-sample (x, y) dicts into batched tensors (the (x, y) contract)."""
import numpy as np
import torch


def align_collate(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]

    x = {"tokenized_sequence": torch.as_tensor(
        np.stack([s["tokenized_sequence"] for s in xs]), dtype=torch.long)}

    y = {}
    for key in ys[0].keys():
        stacked = np.stack([s[key] for s in ys])
        y[key] = torch.as_tensor(stacked, dtype=torch.float32)
    return x, y
