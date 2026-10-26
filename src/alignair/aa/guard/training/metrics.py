"""Binary training diagnostics using NumPy."""
from __future__ import annotations

import numpy as np

from alignair.errors import DataContractError


def binary_auroc(labels, scores) -> float | None:
    """Return AUROC with half credit for ties, or None for a single class."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise DataContractError("AUROC labels and scores must be aligned one-dimensional arrays.")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise DataContractError("AUROC requires binary labels and finite scores.")
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="stable")
    _, starts, counts = np.unique(scores[order], return_index=True, return_counts=True)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.repeat(starts + (counts + 1) / 2, counts)
    auc = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return round(float(auc), 4)
