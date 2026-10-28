"""Categorical probability scores and accuracy by externally supplied allele frequency."""
import numpy as np

from ..core.probabilities import validate_categorical_probabilities, validate_categorical_targets
from ..errors import ConfigurationError, InputError, error_context
from .reports import _aligned
from .statistics import _finite_values


def _observations(probabilities, true_ids):
    probabilities = validate_categorical_probabilities(probabilities)
    return probabilities, validate_categorical_targets(true_ids, probabilities.shape)


def categorical_nll(probabilities, true_ids) -> float:
    """Mean negative log probability of the observed class; clip at 1e-12, no softmax.

    Float16 logarithms are evaluated in float32 so zero probabilities have a finite clipped loss.
    Other dtypes retain their existing NumPy clipping/log/mean arithmetic.
    """
    with error_context(stage="probability_evaluation"):
        probabilities, truth = _observations(probabilities, true_ids)
        selected = probabilities[np.arange(len(truth)), truth]
        if selected.dtype == np.float16:
            selected = selected.astype(np.float32)
        return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def multiclass_brier_score(probabilities, true_ids) -> float:
    """Mean sum of squared class residuals (not divided by class count), without a one-hot matrix."""
    with error_context(stage="probability_evaluation"):
        probabilities, truth = _observations(probabilities, true_ids)
        residual = probabilities.copy(order="K")
        residual[np.arange(len(truth)), truth] -= 1
        np.square(residual, out=residual)
        return float(np.mean(np.sum(residual, axis=1)))


def validate_frequency_edges(edges):
    """Shared bin contract used by frequency reports and scoring preflight; retain caller labels."""
    try:
        values = np.asarray(edges)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("frequency edges must form a numeric vector", context={"field": "freq_edges"}) from exc
    if (values.ndim != 1 or len(values) < 2 or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all() or np.any(np.diff(values.astype(float)) <= 0)
            or values[0] < 0 or values[-1] > 1):
        raise ConfigurationError("frequency edges must be finite, strictly increasing and within [0, 1]",
                                 context={"field": "freq_edges"})


def frequency_accuracy_report(true_ids, predicted_ids, frequencies, edges) -> dict:
    """Accuracy by supplied frequency: left-closed bins, last bin also includes its right edge.

    Frequencies are caller metadata, not estimated from this cohort. Empty bins return acc=None;
    frequencies outside a valid partial edge range are omitted. Edge text retains caller formatting.
    """
    with error_context(stage="frequency_evaluation"):
        truth, predicted = _aligned(true_ids, predicted_ids)
        _finite_values(frequencies, "frequencies")
        frequency = np.asarray(frequencies)  # Keep comparison precision at bin boundaries.
        if (frequency.dtype.kind not in "biuf" or frequency.shape != truth.shape
                or (frequency < 0).any() or (frequency > 1).any()):
            raise InputError("frequencies must be aligned values in [0, 1]", context={"field": "frequencies"})
        validate_frequency_edges(edges)
        out = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            last = hi == edges[-1]
            selected = (frequency >= lo) & (frequency <= hi if last else frequency < hi)
            label = f"[{lo},{hi}]" if last else f"[{lo},{hi})"
            out[label] = {"n": int(selected.sum()),
                          "acc": float((predicted[selected] == truth[selected]).mean()) if selected.any() else None}
        return out
