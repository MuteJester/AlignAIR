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


def categorical_rows(targets: dict, gene: str, logits: torch.Tensor) -> torch.Tensor:
    """Rows with a class label or a nonempty compatible set; -1 means no class supervision."""
    valid = supervised_rows(targets, gene, logits) & _row_mask(targets, f"{gene}_class_supervise", logits)
    key = f"{gene}_compatible"
    if key in targets:
        compatible = targets[key]
        if compatible.shape != logits.shape:
            raise TrainingDataError(f"{key} must match the class-logit shape",
                                    context={"gene": gene, "field": key})
        if not ((compatible[valid] == 0) | (compatible[valid] == 1)).all():
            raise TrainingDataError(f"{key} must contain binary class membership",
                                    context={"gene": gene, "field": key})
        if (valid & ~(compatible != 0).any(-1)).any():
            raise TrainingDataError(f"{key} is empty for a supervised row",
                                    context={"gene": gene, "field": key})
        return valid
    key = f"{gene}_class"
    labels = targets[key]
    if labels.shape != valid.shape or labels.dtype != torch.long:
        raise TrainingDataError(f"{key} must be an integer class-index row vector",
                                context={"gene": gene, "field": key})
    classified = valid & (labels != -1)
    if ((labels[classified] < 0) | (labels[classified] >= logits.shape[1])).any():
        raise TrainingDataError(f"{key} has a class index outside the model vocabulary",
                                context={"gene": gene, "field": key})
    return classified
