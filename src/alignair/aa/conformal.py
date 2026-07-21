"""Split-conformal prediction sets (APS / RAPS) with optional Mondrian (group-conditioned) calibration.

Fit a conformal quantile on a held-out CALIBRATION cohort, then emit prediction SETS on untouched test
data with a finite-sample marginal-coverage guarantee ≈ 1-α. APS = Adaptive Prediction Sets (Romano
et al.); RAPS adds a size penalty. Mondrian fits a separate quantile per group (e.g. coverage bin),
with a pooled fallback when a group is too small. Used to turn the softmax V/J/D heads into calibrated
allele sets whose size grows honestly as evidence shrinks."""
from __future__ import annotations

import numpy as np


def aps_scores(probs: np.ndarray, true: np.ndarray, rng, lam: float = 0.0, k_reg: int = 1) -> np.ndarray:
    """APS/RAPS non-conformity scores for calibration examples. ``lam>0`` gives RAPS (penalizes rank
    beyond ``k_reg``)."""
    n = probs.shape[0]
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank_of = np.argsort(order, axis=1)                       # rank (0-based) of each class
    rt = rank_of[np.arange(n), true]                          # rank of the true class per example
    cum_true = cum[np.arange(n), rt]                          # prob mass up to & incl. the true class
    p_true = probs[np.arange(n), true]
    u = rng.uniform(size=n)
    score = cum_true - u * p_true                             # randomized APS
    if lam > 0:
        score = score + lam * np.maximum(0, rt + 1 - k_reg)   # RAPS size penalty
    return score


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)      # finite-sample correction
    return float(np.quantile(scores, level, method="higher"))


def aps_set(prob_row: np.ndarray, qhat: float, lam: float = 0.0, k_reg: int = 1) -> set:
    """Prediction set for one example: smallest top-ranked prefix whose (penalized) cumulative mass
    reaches ``qhat``."""
    order = np.argsort(-prob_row)
    cum = np.cumsum(prob_row[order])
    if lam > 0:
        cum = cum + lam * np.maximum(0, np.arange(1, len(order) + 1) - k_reg)
    k = int(np.searchsorted(cum, qhat)) + 1                   # include the class that reaches qhat
    return set(order[:min(k, len(order))].tolist())


class Conformal:
    """APS/RAPS calibrator, optionally Mondrian (per-group quantiles + pooled fallback)."""

    def __init__(self, alpha: float = 0.05, lam: float = 0.0, k_reg: int = 1, min_group: int = 200,
                 seed: int = 0):
        self.alpha, self.lam, self.k_reg, self.min_group = alpha, lam, k_reg, min_group
        self.rng = np.random.default_rng(seed)
        self.q_pooled = None
        self.q_group: dict = {}

    def fit(self, probs, true, groups=None):
        s = aps_scores(probs, np.asarray(true), self.rng, self.lam, self.k_reg)
        self.q_pooled = conformal_quantile(s, self.alpha)
        if groups is not None:
            g = np.asarray(groups)
            for key in np.unique(g):
                idx = g == key
                if idx.sum() >= self.min_group:               # else pooled fallback
                    self.q_group[key] = conformal_quantile(s[idx], self.alpha)
        return self

    def predict_set(self, prob_row, group=None) -> set:
        q = self.q_group.get(group, self.q_pooled) if group is not None else self.q_pooled
        return aps_set(prob_row, q, self.lam, self.k_reg)


class FrozenConformal:
    """Deployable calibrator: a fixed qhat loaded from frozen calibration.json. No re-fitting."""

    def __init__(self, qhat: float, lam: float = 0.0, k_reg: int = 1):
        self.qhat, self.lam, self.k_reg = float(qhat), lam, k_reg

    def predict_set(self, prob_row) -> set:
        return aps_set(prob_row, self.qhat, self.lam, self.k_reg)

    @classmethod
    def from_calibration_json(cls, cal, gene, group="full", lam=0.0, k_reg=1):
        return cls(cal["genes"][gene]["calibration_frozen"]["q_mondrian"][group], lam, k_reg)
