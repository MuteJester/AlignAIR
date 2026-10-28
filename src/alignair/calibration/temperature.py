"""Sigmoid confidence temperature scaling and binary top-1 correctness calibration."""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from ..errors import ConfigurationError, InputError, NumericalError


def _failure(kind, message, field):
    return kind(message, context={"stage": "temperature_calibration", "field": field})


def _positive(value, field):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _failure(ConfigurationError, "temperature parameters must be positive finite numbers", field)
    try:
        value = float(value)
    except OverflowError as exc:
        raise _failure(ConfigurationError, "temperature parameters must be finite", field) from exc
    if not np.isfinite(value) or value <= 0:
        raise _failure(ConfigurationError, "temperature parameters must be positive finite numbers", field)
    return value


def _count(value, field):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise _failure(ConfigurationError, "calibration counts must be positive integers", field)
    return int(value)


def _probabilities(values, *, vector=False, allow_empty=False):
    try:
        values = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise _failure(InputError, "probabilities must form a numeric array", "probabilities") from exc
    if values.dtype.kind not in "iuf" or vector and values.ndim != 1 or not allow_empty and values.size == 0:
        raise _failure(InputError, "probabilities have an invalid shape or numeric type", "probabilities")
    if not np.isfinite(values).all():
        raise _failure(NumericalError, "non-finite confidence probabilities", "probabilities")
    if np.any(values < 0) or np.any(values > 1):
        raise _failure(InputError, "confidence probabilities must lie in [0, 1]", "probabilities")
    return values


def _observations(top1_probs, correct, *, allow_empty=False):
    probabilities = _probabilities(top1_probs, vector=True, allow_empty=allow_empty)
    try:
        truth = np.asarray(correct)
    except (TypeError, ValueError) as exc:
        raise _failure(InputError, "correctness must form a binary vector", "correct") from exc
    if (truth.shape != probabilities.shape or truth.dtype.kind not in "biuf"
            or not np.isfinite(truth).all() or not ((truth == 0) | (truth == 1)).all()):
        raise _failure(InputError, "correctness must be an aligned binary vector", "correct")
    return probabilities.astype(np.float64), truth.astype(np.float64)


def _logit(probabilities):
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _sigmoid(logits, temperature):
    # Expected saturation at very small positive temperatures yields finite 0/1 probabilities.
    with np.errstate(over="ignore", under="ignore"):
        return 1.0 / (1.0 + np.exp(-logits / temperature))


def apply_temperature(probs, temperature):
    """Apply sigmoid(logit(p)/temperature), retaining legacy clipping and floating dtype.

    None/1 returns the original input. Positive scaling is monotone; clipping/rounding can
    create ties, and downstream threshold or relative-set membership can change.
    """
    values = _probabilities(probs, allow_empty=True)
    if temperature is None:
        return probs
    temperature = _positive(temperature, "temperature")
    if temperature == 1.0:
        return probs
    dtype = values.dtype if values.dtype.kind == "f" else np.float64
    return _sigmoid(_logit(values), temperature).astype(dtype)


def _nll(logits, correct, temperature):
    probability = np.clip(_sigmoid(logits, temperature), 1e-7, 1 - 1e-7)
    return float(-np.mean(correct * np.log(probability) + (1 - correct) * np.log(1 - probability)))


def fit_temperature(top1_probs, correct, lo: float = .2, hi: float = 3., n: int = 120) -> float:
    """Minimize binary top-1 correctness NLL on the existing log-spaced grid, first tie wins."""
    lo, hi, n = _positive(lo, "lo"), _positive(hi, "hi"), _count(n, "n")
    if lo > hi:
        raise _failure(ConfigurationError, "lower temperature bound must not exceed upper bound", "lo")
    probabilities, truth = _observations(top1_probs, correct)
    logits = _logit(probabilities)
    return float(min(np.geomspace(lo, hi, n), key=lambda temperature: _nll(logits, truth, temperature)))


def ece(top1_probs, correct, n_bins: int = 10) -> float:
    """Equal-width ECE: first bin closed on both ends, later bins open-left/closed-right."""
    n_bins = _count(n_bins, "n_bins")
    probabilities, truth = _observations(top1_probs, correct, allow_empty=True)
    boundaries = np.linspace(0, 1, n_bins + 1)
    error, count = 0., len(probabilities)
    for i in range(n_bins):
        selected = ((probabilities > boundaries[i]) & (probabilities <= boundaries[i + 1]) if i
                    else (probabilities >= 0) & (probabilities <= boundaries[1]))
        if selected.sum():
            error += selected.sum() / count * abs(truth[selected].mean() - probabilities[selected].mean())
    return float(error)
