"""Calibrate the genotype decision thresholds on simulated diploid genotypes.

Aligns each simulated repertoire ONCE (expensive), then sweeps the decision thresholds cheaply
(numpy) via ``decide_gene_calls``, scoring diploid recovery (precision/recall/zygosity). Stratified
by read depth and SHM stratum so the frozen defaults are robust, not tuned to one regime.
"""
from __future__ import annotations

import math
from itertools import product

from .aggregate import align_repertoire
from .airr import to_genotype_set
from .benchmark import recovery, sample_diploid_genotype, simulate_repertoire
from .infer import GenotypeParams, decide_gene_calls

_DEFAULT_GRID = {"present_thr": [0.001, 0.002, 0.004],
                 "min_support": [0.001, 0.003, 0.006],
                 "leakage_cap": [0.3, 0.5, 0.7]}


def build_cases(model, reference, dataconfig, *, seeds=(0, 1), strata=("clean", "moderate"),
                depths=(250,), het_fraction: float = 0.5, device: str = "cpu", batch_size: int = 128):
    """Simulate + ALIGN (once) a set of diploid-genotype repertoires stratified by SHM and depth."""
    cases = []
    for seed in seeds:
        genotype, meta = sample_diploid_genotype(reference, seed, het_fraction=het_fraction)
        for stratum in strata:
            for n in depths:
                seqs, truth = simulate_repertoire(dataconfig, genotype, n, seed, stratum=stratum)
                if not seqs:
                    continue
                aligned = align_repertoire(model, reference, seqs, device=device, batch_size=batch_size)
                cases.append({"aligned": aligned, "truth": truth, "meta": meta,
                              "stratum": stratum, "depth": n, "seed": seed})
    return cases


def _mean(vals):
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else math.nan


def _aggregate(scores):
    p, r, z = _mean([s["allele_precision"] for s in scores]), _mean([s["allele_recall"] for s in scores]), \
        _mean([s["zygosity_acc"] for s in scores])
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "zygosity_acc": z, "f1": f1, "objective": 0.5 * f1 + 0.5 * z}


def evaluate_params(cases, model, reference, params: GenotypeParams):
    scores = []
    for c in cases:
        gcs = decide_gene_calls(c["aligned"], model, reference, params)
        gs, _ = to_genotype_set(gcs, "IGH")
        scores.append(recovery(gs, c["truth"], truth_genotype=c["meta"]))
    return _aggregate(scores)


def calibrate(model, reference, dataconfig, *, grid=None, cases=None, **case_kw):
    """Grid-search the decision thresholds. Returns ``(best, all_results)``; ``best`` is the combo with
    the highest ``objective`` = 0.5·F1 + 0.5·zygosity. Reuse pre-built ``cases`` to avoid re-aligning."""
    grid = grid or _DEFAULT_GRID
    cases = cases if cases is not None else build_cases(model, reference, dataconfig, **case_kw)
    keys = list(grid)
    results = []
    for combo in product(*[grid[k] for k in keys]):
        params = GenotypeParams(**dict(zip(keys, combo)))
        results.append({"params": dict(zip(keys, combo)), **evaluate_params(cases, model, reference, params)})
    best = max(results, key=lambda r: (r["objective"] if not math.isnan(r["objective"]) else -1.0))
    return best, results
