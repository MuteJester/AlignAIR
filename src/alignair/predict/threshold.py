"""Allele selection — the production ``MaxLikelihoodPercentageThreshold`` (TF AlleleSelector).

Relative-to-max filter (NOT cumulative-confidence): keep every allele whose probability is at least
``pct`` of the per-read max, sorted by probability descending, capped at ``cap``. Defaults ``pct=0.1``,
``cap=3`` (the values both TF CLIs pass in practice). The cumulative-confidence variants exist in TF
but are not the shipped path; they can be added to ``SELECTORS`` later if needed.
"""
from __future__ import annotations

import numpy as np

from .state import GeneCall


def max_likelihood_percentage(p: np.ndarray, pct: float = 0.1, cap: int = 3):
    """Return (indices, likelihoods) of selected alleles for one read's probability vector ``p``.

    Selected = ``{i : p_i >= pct * max(p)}``, ordered by ``p`` descending, truncated to ``cap``.
    """
    p = np.asarray(p, dtype=np.float64)
    bar = float(p.max()) * pct
    idx = np.where(p >= bar)[0]
    idx = idx[np.argsort(-p[idx])]           # highest probability first
    if len(idx) > cap:
        idx = idx[:cap]
    return idx, p[idx]


SELECTORS = {"max_likelihood_percentage": max_likelihood_percentage}


def select_alleles(allele_probs: dict, names: dict, pct: float = 0.1, cap: int = 3,
                   selector: str = "max_likelihood_percentage") -> dict:
    """Map per-gene probability matrices to per-read allele calls.

    ``allele_probs``: {gene: [N, C] sigmoid probs}. ``names``: {gene: [C] allele names, index-aligned
    to the model's output head}. Returns {gene: list[GeneCall] of length N}.
    """
    fn = SELECTORS[selector]
    out: dict[str, list[GeneCall]] = {}
    for gene, probs in allele_probs.items():
        gene_names = names[gene]
        calls = []
        for row in np.asarray(probs):
            idx, lk = fn(row, pct, cap)
            calls.append(GeneCall(tuple(gene_names[i] for i in idx),
                                  tuple(float(x) for x in lk)))
        out[gene] = calls
    return out
