"""Pure exact-class and boundary-error reports over aligned, observed cohort vectors."""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..errors import InputError
from .statistics import (DEFAULT_MAX_RESAMPLE_ELEMENTS, _finite_values, bootstrap_mean_ci,
                         paired_diff_ci, wilson_interval)


def _class_ids(values, field):
    try:
        result = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise InputError("class labels must be a vector of observed nonnegative integer IDs",
                         context={"field": field}) from exc
    if result.ndim != 1 or result.size and (result.dtype.kind not in "iu" or (result < 0).any()):
        raise InputError("class labels must be a vector of observed nonnegative integer IDs",
                         context={"field": field})
    return result


def _aligned(true, predicted):
    true, predicted = _class_ids(true, "true_ids"), _class_ids(predicted, "predicted_ids")
    if len(true) != len(predicted):
        raise InputError("truth and predictions must contain the same number of observations")
    return true, predicted


def macro_accuracy(y_true, y_pred) -> float:
    """Mean class accuracy over the true classes present, with equal weight per class."""
    true, predicted = _aligned(y_true, y_pred)
    accuracies = [(predicted[true == label] == label).mean() for label in np.unique(true)]
    return float(np.mean(accuracies)) if accuracies else 0.0


def majority_class_baseline(class_ids) -> float:
    """Best constant exact-class accuracy on this cohort; ties follow first observation order."""
    ids = _class_ids(class_ids, "class_ids").tolist()
    return Counter(ids).most_common(1)[0][1] / len(ids) if ids else 0.0


def _median_baseline(values):
    values = sorted(values)
    if not values:
        return (0.0, 0.0)
    median = values[len(values) // 2]
    return median, sum(abs(value - median) for value in values) / len(values)


def constant_median_mae(values) -> float:
    """Best constant absolute-error baseline; retain the upper median for even cohorts."""
    return _median_baseline(_finite_values(values, "values").tolist())[1]


def classification_report(true_ids, predicted_ids, *, seed: int = 0,
                          max_resample_elements: int = DEFAULT_MAX_RESAMPLE_ELEMENTS) -> dict:
    """Exact-class accuracy and paired majority-baseline lift, using 95% intervals.

    Labels must be observed single-class IDs. Compatible-label membership, multilabel truth,
    row filtering, and cohort selection belong to the caller's task/adapter policy. The majority
    is fitted to this cohort as a descriptive best-constant baseline, not a trained predictor.
    """
    true, predicted = _aligned(true_ids, predicted_ids)
    n = len(true)
    hits = int((true == predicted).sum())
    micro = hits / n if n else 0.0
    macro = macro_accuracy(true, predicted)
    majority = Counter(true.tolist()).most_common(1)[0][0] if n else None
    diffs = [int(t == p) - int(t == majority) for t, p in zip(true, predicted)]
    return {
        "micro_acc": micro, "macro_acc": macro, "micro_acc_ci95": wilson_interval(hits, n),
        "majority_baseline": majority_class_baseline(true),
        "paired_lift": paired_diff_ci(diffs, seed=seed, max_resample_elements=max_resample_elements),
        "macro_over_micro": macro / micro if micro else 0.0,
        "n_distinct_true_classes": len(set(true.tolist())),
    }


def boundary_error_report(targets, absolute_errors, *, seed: int = 0,
                          max_resample_elements: int = DEFAULT_MAX_RESAMPLE_ELEMENTS) -> dict:
    """MAE and paired constant-median improvement on the same observations, with 95% intervals.

    The caller supplies aligned observed targets and already-computed absolute errors in one
    consistent coordinate unit. No nucleotide/residue conversion or supervision masking occurs.
    """
    targets = _finite_values(targets, "targets").tolist()
    errors = _finite_values(absolute_errors, "absolute_errors").tolist()
    if len(targets) != len(errors) or any(error < 0 for error in errors):
        raise InputError("targets and nonnegative absolute errors must be aligned vectors")
    mae = sum(errors) / len(errors) if errors else 0.0
    median, baseline = _median_baseline(targets)
    diffs = [abs(target - median) - error for target, error in zip(targets, errors)]
    return {
        "mae": mae,
        "mae_ci95": bootstrap_mean_ci(errors, seed=seed, max_resample_elements=max_resample_elements),
        "constant_median_baseline_mae": baseline,
        "paired_improvement": paired_diff_ci(diffs, seed=seed, max_resample_elements=max_resample_elements),
        "relative_reduction": (baseline - mae) / baseline if baseline else 0.0,
    }
