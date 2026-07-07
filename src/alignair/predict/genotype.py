"""Genotype-based likelihood correction (conditional; only when a custom genotype is supplied).

Drops non-genotype alleles (zeros their probability) and redistributes their aggregate mass onto the
genotype alleles *proportionally to each one's own likelihood*, clipped at 1.0 (TF bounded
redistribution). Zeroed alleles never clear the downstream threshold, so this matches TF's
"remove from the vector" behavior while keeping array shapes stable.
"""
from __future__ import annotations

import numpy as np

from .state import Predictions


def adjust_for_genotype(preds: Predictions, genotype: dict, reference) -> Predictions:
    for gene, allowed in genotype.items():
        if gene not in preds.allele:
            continue
        names = reference.gene(gene.upper()).names
        keep = np.array([n in allowed for n in names], dtype=bool)
        probs = preds.allele[gene].copy()
        for row in probs:
            total_geno = row[keep].sum()
            total_non = row[~keep].sum()
            if total_geno > 0:
                row[keep] = np.minimum(1.0, row[keep] + row[keep] * (total_non / total_geno))
            row[~keep] = 0.0
        preds.allele[gene] = probs
    return preds
