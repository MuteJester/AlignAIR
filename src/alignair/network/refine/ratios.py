"""The per-base evidence the refine stages read: a read base against a germline base, as the log-ratio of the base
being that germline base at the read's mutation rate m against its being an N base."""
from __future__ import annotations

import torch

RATE = (0.01, 0.25)       # the mutation rate the ratios use, clamped
OUTSIDE = -8.0            # the ratio of a read base paired with no germline base


def pair_ratios(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(match, mismatch)`` ratios at the mutation rate ``m``: ``log(4(1-m))`` and ``log(4m/3)``."""
    return torch.log(4 * (1 - m)), torch.log(4 * m / 3)
