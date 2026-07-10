"""Genotype-recovery benchmark: simulate a repertoire from a KNOWN genotype, then score recovery.

Shaped for a later head-to-head vs TIgGER on the same simulated repertoires.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .infer import _gene_of

_ZYGOSITY_COUNT = {"homozygous": 1, "heterozygous": 2, "duplication": 3}


def sample_diploid_genotype(reference, seed: int, *, genes=("v", "d", "j"), het_fraction: float = 0.5,
                            deletion_fraction: float = 0.0, duplication_fraction: float = 0.0):
    """A biologically-realistic genotype: ≤2 alleles per gene (homozygous/heterozygous), unless the
    deletion/duplication knobs are set (for explicitly testing CNV). Returns
    ``(genotype={gene_type: set(alleles)}, meta={gene_name: zygosity})``."""
    rng = random.Random(seed)
    genotype, meta = {}, {}
    for gtype in genes:
        by_gene: dict[str, list] = defaultdict(list)
        for name in reference.gene(gtype.upper()).names:
            by_gene[_gene_of(name)].append(name)
        allowed: set = set()
        for gene, alleles in by_gene.items():
            u = rng.random()
            if u < deletion_fraction:
                meta[gene] = "deleted"
                continue
            if u < deletion_fraction + duplication_fraction and len(alleles) >= 3:
                chosen, meta[gene] = rng.sample(alleles, 3), "duplication"
            elif len(alleles) >= 2 and rng.random() < het_fraction:
                chosen, meta[gene] = rng.sample(alleles, 2), "heterozygous"
            else:
                chosen, meta[gene] = rng.sample(alleles, 1), "homozygous"
            allowed.update(chosen)
        genotype[gtype] = allowed
    return genotype, meta


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


def recovery(genotype_set: dict, truth_records: list, *, truth_genotype: dict | None = None) -> dict:
    """Precision/recall of the inferred documented alleles vs the alleles actually used in ``truth``.
    ``truth_genotype`` (``{gene_name: zygosity}``, from :func:`sample_diploid_genotype`) gives EXACT
    zygosity accuracy + deletion recall; without it, zygosity is estimated from the used allele counts."""
    truth_all: set = set()
    for r in truth_records:
        for g in ("v", "d", "j"):
            for a in str(r.get(f"{g}_call") or "").split(","):
                if a.strip():
                    truth_all.add(a.strip())
    inferred: set = set()
    inferred_del: set = set()
    for gt in genotype_set.get("genotype_class_list", []):
        for da in gt.get("documented_alleles", []):
            inferred.add(da["label"])
        for dg in gt.get("deleted_genes", []):
            inferred_del.add(dg["label"])

    tp = inferred & truth_all
    out = {"n_truth_alleles": len(truth_all), "n_inferred": len(inferred),
           "allele_precision": len(tp) / len(inferred) if inferred else math.nan,
           "allele_recall": len(tp) / len(truth_all) if truth_all else math.nan}

    inf_by_gene: dict[str, set] = defaultdict(set)
    for a in inferred:
        inf_by_gene[_gene_of(a)].add(a)

    if truth_genotype is not None:                             # exact zygosity + deletion vs the intended genotype
        n_correct = n_scored = 0
        for gene, z in truth_genotype.items():
            if z == "deleted":
                continue
            if gene in inf_by_gene:
                n_scored += 1
                n_correct += len(inf_by_gene[gene]) == _ZYGOSITY_COUNT.get(z, len(inf_by_gene[gene]))
        out["zygosity_acc"] = n_correct / n_scored if n_scored else math.nan
        truth_del = {g for g, z in truth_genotype.items() if z == "deleted"}
        out["deletion_recall"] = len(inferred_del & truth_del) / len(truth_del) if truth_del else math.nan
    else:                                                      # estimate zygosity from used allele counts
        truth_by_gene: dict[str, set] = defaultdict(set)
        for a in truth_all:
            truth_by_gene[_gene_of(a)].add(a)
        common = set(truth_by_gene) & set(inf_by_gene)
        out["zygosity_acc"] = (sum(1 for g in common if len(truth_by_gene[g]) == len(inf_by_gene[g])) / len(common)
                               if common else math.nan)
    return out
