"""Allele selection — turn per-allele probabilities into a called allele *set*.

The chip's allele scores are each allele's likelihood relative to the read's best allele (the bank scan's
posterior divided by its maximum: the best allele scores 1). The set keeps every allele scoring at least the
threshold (``"absolute"``, the default, at 0.5).

Alternatives: ``"largest_gap"`` (parameter-free: cut at the biggest drop in the sorted scores) and
``"max_likelihood_percentage"`` (a cut relative to the top score).

Every selector takes ``(p, param, cap)`` and returns ``(indices, likelihoods)`` sorted by
probability, non-empty (always keeps top-1), and capped at ``cap``.
"""
from __future__ import annotations

import numpy as np

from .state import GeneCall


def _finish(p, idx, cap):
    idx = idx[np.argsort(-p[idx])]
    return idx[:cap], p[idx[:cap]]


def absolute_threshold(p: np.ndarray, thr: float = 0.5, cap: int = 3):
    """Calibrated-posterior rule: keep alleles with ``p >= thr`` (default 0.5). Derived, no fit."""
    p = np.asarray(p, dtype=np.float64)
    idx = np.where(p >= thr)[0]
    if len(idx) == 0:
        idx = np.array([int(p.argmax())])            # never empty -> keep top-1
    return _finish(p, idx, cap)


def largest_gap(p: np.ndarray, param=None, cap: int = 3):
    """Parameter-free: keep the top cluster, cutting at the largest drop among the top ``cap+1``."""
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p)[: cap + 1]
    sp = p[order]
    k = int(np.argmax(sp[:-1] - sp[1:])) + 1 if len(sp) > 1 else 1
    idx = order[:k]
    return idx, p[idx]


def max_likelihood_percentage(p: np.ndarray, pct: float = 0.1, cap: int = 3):
    """The relative rule: keep ``{i : p_i >= pct * max(p)}``."""
    p = np.asarray(p, dtype=np.float64)
    idx = np.where(p >= float(p.max()) * pct)[0]
    return _finish(p, idx, cap)


SELECTORS = {"absolute": absolute_threshold, "largest_gap": largest_gap,
             "max_likelihood_percentage": max_likelihood_percentage}


def select_alleles(allele_probs: dict, names: dict, param: float = 0.5, cap: int = 3,
                   selector: str = "absolute") -> dict:
    """Map per-gene score matrices to per-read allele calls.

    ``allele_probs``: {gene: [N, C] allele scores}. ``names``: {gene: [C] allele names, index-aligned to the bound
    bank}. ``param`` is the selector's scalar (threshold for ``"absolute"``, a percentage for the relative rule,
    ignored for ``"largest_gap"``). Returns {gene: list[GeneCall]}. A genotype restricts the calls by being the bank.
    """
    fn = SELECTORS[selector]
    out: dict[str, list[GeneCall]] = {}
    for gene, probs in allele_probs.items():
        gene_names = names[gene]
        calls = []
        for row in np.asarray(probs):
            idx, lk = fn(row, param, cap)
            calls.append(GeneCall(tuple(gene_names[j] for j in idx), tuple(float(x) for x in lk)))
        out[gene] = calls
    return out
