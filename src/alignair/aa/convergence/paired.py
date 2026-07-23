"""Paired-statistics module for model-comparison confirmation studies.

Pure functions over per-read top-1 correctness arrays. `a` denotes the
baseline/incumbent, `b` the candidate/winner; `a_correct`/`b_correct` are
boolean arrays of per-read correctness on the SAME paired reads (i.e. index i
in both arrays refers to the same read).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def paired_bootstrap_diff_ci(a_correct, b_correct, n_boot=2000, seed=0, alpha=0.05) -> dict:
    """Paired bootstrap over reads for mean(b) - mean(a).

    Resamples read indices with replacement (the same indices applied to
    both a and b on each resample) and computes mean(b[idx]) - mean(a[idx])
    per resample.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: len(a_correct)={len(a)} != len(b_correct)={len(b)}")
    if len(a) == 0:
        raise ValueError("a_correct/b_correct must be non-empty")

    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)

    return {
        "diff": float(b.mean() - a.mean()),
        "lo": float(np.percentile(diffs, alpha / 2 * 100)),
        "hi": float(np.percentile(diffs, (1 - alpha / 2) * 100)),
        "n": n,
    }


def mcnemar(a_correct, b_correct) -> dict:
    """Exact McNemar test on discordant pairs between a and b.

    b_gain = reads a got wrong that b gets right (winner fixes).
    b_loss = reads a got right that b gets wrong (winner breaks).
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: len(a_correct)={len(a)} != len(b_correct)={len(b)}")

    b_gain = int((~a & b).sum())
    b_loss = int((a & ~b).sum())
    n_discordant = b_gain + b_loss

    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = min(1.0, binomtest(min(b_gain, b_loss), n=n_discordant, p=0.5).pvalue)

    return {
        "b_gain": b_gain,
        "b_loss": b_loss,
        "p_value": float(p_value),
        "net": int(b_gain - b_loss),
    }


def class_stratified_bootstrap_macro_diff(
    y_true, a_correct, b_correct, n_boot=2000, seed=0, alpha=0.05
) -> dict:
    """Bootstrap CI for the macro-averaged (per-class mean correctness) diff.

    Macro = mean over classes of per-class mean correctness. The point
    estimate is computed over all classes present in y_true. The bootstrap
    resamples the set of unique classes with replacement; for each resampled
    class list, macro_a/macro_b are recomputed over those classes (using
    each class's per-read correctness), and the diff is recorded.
    """
    y = np.asarray(y_true)
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if not (len(y) == len(a) == len(b)):
        raise ValueError(
            f"length mismatch: len(y_true)={len(y)}, len(a_correct)={len(a)}, len(b_correct)={len(b)}"
        )
    if len(y) == 0:
        raise ValueError("y_true/a_correct/b_correct must be non-empty")

    classes = np.unique(y)
    per_class_a = np.array([a[y == c].mean() for c in classes])
    per_class_b = np.array([b[y == c].mean() for c in classes])

    diff = float(per_class_b.mean() - per_class_a.mean())

    n_classes = len(classes)
    rng = np.random.default_rng(seed)
    cls_idx = rng.integers(0, n_classes, size=(n_boot, n_classes))
    boot_a = per_class_a[cls_idx].mean(axis=1)
    boot_b = per_class_b[cls_idx].mean(axis=1)
    diffs = boot_b - boot_a

    return {
        "diff": diff,
        "lo": float(np.percentile(diffs, alpha / 2 * 100)),
        "hi": float(np.percentile(diffs, (1 - alpha / 2) * 100)),
    }
