"""Allele selection: turn per-allele scores into a called allele *set*.

The chip's allele scores are each allele's likelihood relative to the read's best allele (the bank scan's
posterior divided by its maximum: the best allele scores 1). The set keeps every allele scoring at least the
threshold (0.5 by default), best first, at most ``cap`` of them, and never empty (the best allele always stays).
"""
from __future__ import annotations

import numpy as np

from .state import GeneCall


def _kept(p: np.ndarray, threshold: float, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """The indices of the alleles scoring at least ``threshold`` (the best one if none does), best first and at most
    ``cap``, with their scores."""
    p = np.asarray(p, dtype=np.float64)
    idx = np.where(p >= threshold)[0]
    if len(idx) == 0:
        idx = np.array([int(p.argmax())])
    idx = idx[np.argsort(-p[idx])][:cap]
    return idx, p[idx]


def select_alleles(allele_probs: dict[str, np.ndarray], names: dict[str, list[str]], threshold: float = 0.5,
                   cap: int = 3) -> dict[str, list[GeneCall]]:
    """Map per-gene score matrices to per-read allele calls.

    ``allele_probs``: {gene: [N, C] allele scores}. ``names``: {gene: [C] allele names, index-aligned to the bound
    bank}. Returns {gene: list[GeneCall]}. A genotype restricts the calls by being the bank.
    """
    out: dict[str, list[GeneCall]] = {}
    for gene, probs in allele_probs.items():
        gene_names = names[gene]
        calls = []
        for row in np.asarray(probs):
            idx, lk = _kept(row, threshold, cap)
            calls.append(GeneCall(tuple(gene_names[j] for j in idx), tuple(float(x) for x in lk)))
        out[gene] = calls
    return out
