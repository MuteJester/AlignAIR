"""Identifiability audit: how many alleles survive translation as uniquely distinguishable,
full-length and under coverage-limited windows (truncated reads collapse classes further)."""
from __future__ import annotations

from .equivalence import EquivalenceClassMap


def identifiability_summary(cmap: EquivalenceClassMap) -> dict:
    sizes = [len(m) for m in cmap.class_members.values()]
    n_alleles = len(cmap.allele_class)
    n_unique = sum(1 for s in sizes if s == 1)
    hist: dict[int, int] = {}
    for s in sizes:
        hist[s] = hist.get(s, 0) + 1
    return {
        "gene": cmap.gene,
        "n_alleles": n_alleles,
        "n_classes": len(sizes),
        "n_unique_alleles": n_unique,
        "frac_alleles_uniquely_identifiable": (n_unique / n_alleles) if n_alleles else 0.0,
        "ambiguity_size_hist": {str(k): hist[k] for k in sorted(hist)},
        "max_ambiguity": max(sizes) if sizes else 0,
    }


def window_identifiability(aa_by_name: dict, lo: int, hi: int) -> dict:
    """Group alleles by ``aa_seq[lo:hi]``; a covered sub-window merges classes that agree on it."""
    groups: dict[str, list] = {}
    for name, aa in aa_by_name.items():
        groups.setdefault(aa[lo:hi], []).append(name)
    n = len(aa_by_name)
    n_unique = sum(1 for m in groups.values() if len(m) == 1)
    return {
        "lo": lo, "hi": hi, "n_alleles": n,
        "n_classes": len(groups),
        "frac_uniquely_identifiable": (n_unique / n) if n else 0.0,
    }


def prefix_identifiability_curve(aa_by_name: dict, lengths: list) -> list:
    """Identifiability seeing only the first ``k`` residues, for each ``k`` in ``lengths``."""
    return [window_identifiability(aa_by_name, 0, k) for k in lengths]
