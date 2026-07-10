"""Genotype-recovery benchmark: simulate a repertoire from a KNOWN genotype, then score recovery.

Shaped for a later head-to-head vs TIgGER on the same simulated repertoires.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .infer import _gene_of


def _accept(rec: dict, genotype: dict) -> bool:
    for g, allowed in genotype.items():
        if not allowed:
            continue
        calls = [a.strip() for a in str(rec.get(f"{g}_call") or "").split(",") if a.strip()]
        if not calls or not all(a in allowed for a in calls):
            return False
    return True


def simulate_repertoire(dataconfig, genotype: dict, n: int, seed: int, *, stratum: str = "moderate",
                        gen_batch: int = 300):
    """``n`` reads whose true V/D/J alleles all lie in ``genotype`` (rejection-sampled from the gym)."""
    from ..evaluate.benchmark import default_strata, generate_labeled
    kept, s = [], seed
    while len(kept) < n and s < seed + 200:
        for r in generate_labeled(dataconfig, default_strata()[stratum], gen_batch, s):
            if _accept(r, genotype):
                kept.append(r)
                if len(kept) >= n:
                    break
        s += 1
    kept = kept[:n]
    return [r["sequence"] for r in kept], kept


def recovery(genotype_set: dict, truth_records: list) -> dict:
    """Precision/recall of the inferred documented alleles vs the alleles actually used in ``truth``,
    plus per-gene zygosity accuracy (matching allele counts)."""
    truth_all: set = set()
    for r in truth_records:
        for g in ("v", "d", "j"):
            for a in str(r.get(f"{g}_call") or "").split(","):
                if a.strip():
                    truth_all.add(a.strip())
    inferred: set = set()
    for gt in genotype_set.get("genotype_class_list", []):
        for da in gt.get("documented_alleles", []):
            inferred.add(da["label"])

    tp = inferred & truth_all
    precision = len(tp) / len(inferred) if inferred else math.nan
    recall = len(tp) / len(truth_all) if truth_all else math.nan

    truth_by_gene, inf_by_gene = defaultdict(set), defaultdict(set)
    for a in truth_all:
        truth_by_gene[_gene_of(a)].add(a)
    for a in inferred:
        inf_by_gene[_gene_of(a)].add(a)
    common = set(truth_by_gene) & set(inf_by_gene)
    zyg = (sum(1 for g in common if len(truth_by_gene[g]) == len(inf_by_gene[g])) / len(common)
           if common else math.nan)

    return {"n_truth_alleles": len(truth_all), "n_inferred": len(inferred),
            "allele_precision": precision, "allele_recall": recall, "zygosity_acc": zyg}
