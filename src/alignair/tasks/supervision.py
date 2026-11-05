"""Shared row-supervision rules for alignment losses and metrics."""
from __future__ import annotations

import torch

from ..errors import TrainingDataError


def _row_mask(targets, key, reference):
    shape = (reference.shape[0],)
    if key not in targets:
        return torch.ones(shape, dtype=torch.bool, device=reference.device)
    mask = targets[key]
    if not isinstance(mask, torch.Tensor) or mask.shape != shape or not ((mask == 0) | (mask == 1)).all():
        raise TrainingDataError(f"{key} must be a binary row mask of shape {shape}",
                                context={"gene": key.split("_", 1)[0], "field": key})
    return mask.to(device=reference.device, dtype=torch.bool)


def supervised_rows(targets: dict, gene: str, reference: torch.Tensor) -> torch.Tensor:
    """Combine binary gene masks; masks are row vectors, never positional labels."""
    return (_row_mask(targets, f"{gene}_supervise", reference)
            & _row_mask(targets, f"{gene}_valid", reference))
