"""Frozen held-out evaluation with explicit baselines and uncertainty.

A raw accuracy is uninterpretable: 149 V classes are highly imbalanced, so class accuracy can
reflect allele-frequency priors, and boundary MAE can be beaten by a constant. Every metric here
is reported against its best-constant baseline, compared with a PAIRED bootstrap (the model and
the baseline predict the same reads, so independent intervals are the wrong test)."""
from __future__ import annotations

import math
from collections import Counter

import torch

from .data import collate_aa


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Descriptive interval for a single proportion. NOT the gate statistic (see paired_diff_ci)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values, n_boot: int = 1000, seed: int = 0):
    import numpy as np
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_diff_ci(diffs, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap CI of the MEAN PER-ITEM DIFFERENCE (model advantage).

    The model and the baseline predict the SAME reads, so their errors are correlated; comparing
    two independent Wilson/bootstrap intervals is the wrong test and is far too conservative.
    The gate statistic is this paired difference with its lower bound above zero."""
    lo, hi = bootstrap_mean_ci(diffs, n_boot=n_boot, seed=seed)
    mean = (sum(diffs) / len(diffs)) if len(diffs) else 0.0
    return {"mean": mean, "ci95": (lo, hi), "lower_gt_zero": lo > 0.0}


def majority_class_baseline(class_ids) -> float:
    ids = list(class_ids)
    if not ids:
        return 0.0
    return Counter(ids).most_common(1)[0][1] / len(ids)


def constant_median_mae(values) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    med = vals[len(vals) // 2]
    return sum(abs(v - med) for v in vals) / len(vals)


def _macro_accuracy(true_ids, pred_ids) -> float:
    per: dict = {}
    for t, p in zip(true_ids, pred_ids):
        hit, n = per.get(t, (0, 0))
        per[t] = (hit + int(t == p), n + 1)
    return (sum(h / n for h, n in per.values()) / len(per)) if per else 0.0


@torch.no_grad()
def evaluate_frozen(model, samples, cfg, class_maps, *, batch_size: int = 128,
                    device: str = "cpu") -> dict:
    """Evaluate over the WHOLE frozen held-out set in batches, with baselines + intervals."""
    model.eval()
    true = {"v": [], "j": []}
    pred = {"v": [], "j": []}
    errs = {f"{g}_{b}": [] for g in ("v", "j") for b in ("start", "end")}
    tgts = {f"{g}_{b}": [] for g in ("v", "j") for b in ("start", "end")}
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
    for g in ("v", "j"):
        n = len(true[g])
        hits = sum(int(t == p) for t, p in zip(true[g], pred[g]))
        micro = hits / n if n else 0.0
        macro = _macro_accuracy(true[g], pred[g])
        # PAIRED comparison: the majority predictor scores the same reads.
        maj_class = Counter(true[g]).most_common(1)[0][0] if n else None
        diffs = [int(t == p) - int(t == maj_class) for t, p in zip(true[g], pred[g])]
        res["classification"][g] = {
            "micro_acc": micro, "macro_acc": macro,
            "micro_acc_ci95": wilson_interval(hits, n),          # descriptive only
            "majority_baseline": majority_class_baseline(true[g]),
            "paired_lift": paired_diff_ci(diffs),                # <- the gate statistic
            "macro_over_micro": (macro / micro) if micro else 0.0,
            "n_distinct_true_classes": len(set(true[g])),
        }
    for key, e in errs.items():
        mae = sum(e) / len(e) if e else 0.0
        base_mae = constant_median_mae(tgts[key])
        med = sorted(tgts[key])[len(tgts[key]) // 2] if tgts[key] else 0.0
        # PAIRED improvement: constant-predictor error minus model error, per read.
        diffs = [abs(t - med) - me for t, me in zip(tgts[key], e)]
        res["boundaries"][key] = {
            "mae": mae, "mae_ci95": bootstrap_mean_ci(e),
            "constant_median_baseline_mae": base_mae,
            "paired_improvement": paired_diff_ci(diffs),         # <- the gate statistic
            "relative_reduction": ((base_mae - mae) / base_mae) if base_mae else 0.0,
        }

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
