"""Protein-study frozen-cohort inference and coverage/ambiguity strata.

Reusable statistics and report calculations live in alignair.evaluate. This study adapter retains
its V/J exact-class policy, categorical batch construction and protein-specific strata.
"""
from __future__ import annotations

from ..data.adapters import collate_categorical as collate_aa
from ..train.engine import evaluation_mode
from ..evaluate.reports import (boundary_error_report, classification_report,
                                constant_median_mae, majority_class_baseline,
                                macro_accuracy as _macro_accuracy)
from ..evaluate.statistics import bootstrap_mean_ci, paired_diff_ci, wilson_interval


def evaluate_frozen(model, samples, cfg, class_maps, *, batch_size: int = 128,
                    device: str = "cpu") -> dict:
    """Evaluate over the WHOLE frozen held-out set in batches, with baselines + intervals."""
    true = {"v": [], "j": []}
    pred = {"v": [], "j": []}
    errs = {f"{g}_{b}": [] for g in ("v", "j") for b in ("start", "end")}
    tgts = {f"{g}_{b}": [] for g in ("v", "j") for b in ("start", "end")}
    with evaluation_mode(model):
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            bi, tg = collate_aa(chunk, cfg.max_seq_length, device)
            out = model(bi)
            for g in ("v", "j"):
                true[g] += tg[f"{g}_class"].tolist()
                pred[g] += out[f"{g}_class_logits"].argmax(-1).tolist()
                for b in ("start", "end"):
                    e = (out[f"{g}_{b}"].squeeze(-1) - tg[f"{g}_{b}"].squeeze(-1)).abs().tolist()
                    errs[f"{g}_{b}"] += e
                    tgts[f"{g}_{b}"] += tg[f"{g}_{b}"].squeeze(-1).tolist()

    res: dict = {"n": len(samples), "classification": {}, "boundaries": {}, "stratified": {}}
    for gene in ("v", "j"):
        res["classification"][gene] = classification_report(true[gene], pred[gene])
    for key, errors in errs.items():
        res["boundaries"][key] = boundary_error_report(tgts[key], errors)

    # V accuracy stratified by observed coverage and by protein-class ambiguity size (DIAGNOSTIC)
    cov_buckets, amb_buckets = {}, {}
    vmap = class_maps["V"]
    for s, t, p in zip(samples, true["v"], pred["v"]):
        cov = s["v_end"] - s["v_start"]
        cb = "<40" if cov < 40 else ("40-79" if cov < 80 else ">=80")
        hit, n = cov_buckets.get(cb, (0, 0))
        cov_buckets[cb] = (hit + int(t == p), n + 1)
        amb = len(vmap.ambiguity_set(t))
        ab = str(amb) if amb < 3 else ">=3"
        hit, n = amb_buckets.get(ab, (0, 0))
        amb_buckets[ab] = (hit + int(t == p), n + 1)
    res["stratified"]["v_acc_by_coverage"] = {
        k: {"acc": h / n, "n": n, "ci95": wilson_interval(h, n)}
        for k, (h, n) in sorted(cov_buckets.items())}
    res["stratified"]["v_acc_by_ambiguity_size"] = {
        k: {"acc": h / n, "n": n, "ci95": wilson_interval(h, n)}
        for k, (h, n) in sorted(amb_buckets.items())}
    return res
