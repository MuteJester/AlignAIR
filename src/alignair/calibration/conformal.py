"""APS/RAPS prediction sets with pooled and group-conditioned calibration thresholds.

The existing randomized score, clipped higher-quantile convention, NumPy ranking and
deterministic boundary-inclusive prediction prefix are retained for artifact compatibility.
Calibration cohorts, distribution assumptions and label identity belong to the caller.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from ..errors import ConfigurationError, InputError, NumericalError, error_context
from ..core.probabilities import validate_categorical_probabilities, validate_categorical_targets


def _failure(kind, message, field):
    return kind(message, context={"stage": "conformal_calibration", "field": field})


def _real(value, field):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _failure(ConfigurationError, "calibration parameters must be real numbers", field)
    try:
        value = float(value)
    except OverflowError as exc:
        raise _failure(ConfigurationError, "calibration parameters must be finite", field) from exc
    if not np.isfinite(value) or value < 0:
        raise _failure(ConfigurationError, "calibration parameters must be finite and nonnegative", field)
    return value


def _integer(value, field, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise _failure(ConfigurationError, f"{field} must be an integer >= {minimum}", field)
    return int(value)


def _alpha(value):
    value = _real(value, "alpha")
    if not 0 < value < 1:
        raise _failure(ConfigurationError, "alpha must lie strictly between zero and one", "alpha")
    return value


def _array(values, dimensions, field):
    try:
        values = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise _failure(InputError, "calibration values must form a numeric array", field) from exc
    if values.ndim != dimensions or not all(values.shape) or values.dtype.kind not in "iuf":
        raise _failure(InputError, "calibration array has an invalid shape or numeric type", field)
    if not np.isfinite(values).all():
        raise _failure(NumericalError, "non-finite calibration values", field)
    return values


def _groups(values, count):
    try:
        values = np.asarray(values)
        if values.shape != (count,) or any(
                key is None or isinstance(key, Real) and not np.isfinite(key) for key in values):
            raise ValueError
        keys = np.unique(values)
        for key in keys:
            hash(key)
    except (TypeError, ValueError) as exc:
        raise _failure(InputError, "groups must be aligned, comparable, non-null finite scalar labels", "groups") from exc
    return values, keys


def _aps_scores(probs, true, rng, lam, k_reg):
    n = probs.shape[0]
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank_of = np.argsort(order, axis=1)
    rt = rank_of[np.arange(n), true]
    cum_true = cum[np.arange(n), rt]
    p_true = probs[np.arange(n), true]
    u = rng.uniform(size=n)
    score = cum_true - u * p_true
    if lam > 0:
        score = score + lam * np.maximum(0, rt + 1 - k_reg)
    return score


def aps_scores(probs, true, rng, lam: float = 0.0, k_reg: int = 1) -> np.ndarray:
    """Randomized APS scores, optionally adding lambda times rank beyond k_reg (RAPS)."""
    with error_context(stage="conformal_calibration"):
        probs = validate_categorical_probabilities(probs)
        true = validate_categorical_targets(true, probs.shape)
    lam, k_reg = _real(lam, "lam"), _integer(k_reg, "k_reg", 0)
    if not isinstance(rng, np.random.Generator):
        raise _failure(ConfigurationError, "APS scoring requires a NumPy Generator", "rng")
    return _aps_scores(probs, true, rng, lam, k_reg)


def conformal_quantile(scores, alpha: float) -> float:
    """Retain min(1, ceil((n+1)*(1-alpha))/n) with NumPy's higher interpolation."""
    scores, alpha = _array(scores, 1, "scores"), _alpha(alpha)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def _prediction_set(prob_row, qhat, lam, k_reg, class_count=None):
    with error_context(stage="conformal_calibration"):
        prob_row = validate_categorical_probabilities(prob_row, 1)
    if class_count is not None and len(prob_row) != class_count:
        raise _failure(InputError, "prediction width differs from the fitted class vocabulary", "probabilities")
    qhat, lam, k_reg = _real(qhat, "qhat"), _real(lam, "lam"), _integer(k_reg, "k_reg", 0)
    order = np.argsort(-prob_row)
    cum = np.cumsum(prob_row[order])
    if lam > 0:
        cum = cum + lam * np.maximum(0, np.arange(1, len(order) + 1) - k_reg)
    k = int(np.searchsorted(cum, qhat)) + 1
    return set(order[:min(k, len(order))].tolist())


def aps_set(prob_row, qhat: float, lam: float = 0.0, k_reg: int = 1) -> set:
    """Boundary-inclusive top-ranked prefix; a threshold above all scores yields every class."""
    return _prediction_set(prob_row, qhat, lam, k_reg)


class Conformal:
    """Fit APS/RAPS thresholds, optionally per group with pooled fallback.

    A successful refit replaces all thresholds. Input validation precedes random score draws;
    thresholds are replaced only after all quantiles are computed. The local generator advances
    across fits, while prediction itself draws no random values.
    """

    def __init__(self, alpha: float = 0.05, lam: float = 0.0, k_reg: int = 1, min_group: int = 200,
                 seed: int = 0):
        self.alpha, self.lam = _alpha(alpha), _real(lam, "lam")
        self.k_reg, self.min_group = _integer(k_reg, "k_reg", 0), _integer(min_group, "min_group", 1)
        try:
            self.rng = np.random.default_rng(seed)
        except (TypeError, ValueError) as exc:
            raise _failure(ConfigurationError, "invalid calibration generator seed", "seed") from exc
        self.q_pooled = None
        self.q_group: dict = {}
        self.class_count = None

    def fit(self, probs, true, groups=None):
        with error_context(stage="conformal_calibration"):
            probs = validate_categorical_probabilities(probs)
            true = validate_categorical_targets(true, probs.shape)
        alpha, lam = _alpha(self.alpha), _real(self.lam, "lam")
        k_reg, min_group = _integer(self.k_reg, "k_reg", 0), _integer(self.min_group, "min_group", 1)
        grouped = None if groups is None else _groups(groups, len(probs))
        scores = _aps_scores(probs, true, self.rng, lam, k_reg)
        pooled, thresholds = conformal_quantile(scores, alpha), {}
        if grouped is not None:
            values, keys = grouped
            for key in keys:
                selected = values == key
                if selected.sum() >= min_group:
                    thresholds[key] = conformal_quantile(scores[selected], alpha)
        self.q_pooled, self.q_group, self.class_count = pooled, thresholds, probs.shape[1]
        return self

    def predict_set(self, prob_row, group=None) -> set:
        if self.q_pooled is None:
            raise _failure(ConfigurationError, "fit calibration before predicting sets", "fit")
        try:
            qhat = self.q_group.get(group, self.q_pooled) if group is not None else self.q_pooled
        except TypeError as exc:
            raise _failure(InputError, "prediction group must be a hashable scalar label", "group") from exc
        return _prediction_set(prob_row, qhat, self.lam, self.k_reg, self.class_count)


class FrozenConformal:
    """Apply a stored threshold without fitting; artifact/label binding remains with the loader."""

    def __init__(self, qhat: float, lam: float = 0.0, k_reg: int = 1):
        self.qhat, self.lam, self.k_reg = _real(qhat, "qhat"), _real(lam, "lam"), _integer(k_reg, "k_reg", 0)

    def predict_set(self, prob_row) -> set:
        return aps_set(prob_row, self.qhat, self.lam, self.k_reg)

    @classmethod
    def from_calibration_json(cls, cal, gene, group="full", lam=0.0, k_reg=1):
        return cls(cal["genes"][gene]["calibration_frozen"]["q_mondrian"][group], lam, k_reg)
