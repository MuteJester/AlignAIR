"""Post-hoc confidence calibration for the allele heads (temperature scaling).

The V/D/J heads are multi-label sigmoid trained with BCE label-smoothing (0.1), so the top-1 allele
probability systematically *understates* accuracy (under-confident). A single per-gene temperature
``T`` rescales the sigmoid logits — ``sigmoid(logit(p)/T)`` — to align confidence with accuracy.
``T<1`` sharpens (more confident), ``T>1`` softens; the transform is monotonic, so the argmax / call
set is unchanged (accuracy is preserved, only the reported likelihoods move). Fit ``T`` by minimizing
the NLL of top-1 correctness on held-out reads (Guo et al. 2017), evaluate with ECE.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def apply_temperature(probs, T):
    """Rescale sigmoid probabilities by temperature ``T``: ``sigmoid(logit(p)/T)``. Monotonic, so
    ranking/argmax is preserved. ``T`` None/1.0 -> unchanged."""
    if not T or T == 1.0:
        return probs
    return (1.0 / (1.0 + np.exp(-_logit(probs) / T))).astype(np.asarray(probs).dtype)


def _nll(z, y, T):
    p = np.clip(1.0 / (1.0 + np.exp(-z / T)), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(top1_probs, correct, lo: float = 0.2, hi: float = 3.0, n: int = 120) -> float:
    """Temperature minimizing NLL of top-1 correctness (1-D grid search over a log-spaced range)."""
    z = _logit(top1_probs)
    y = np.asarray(correct, dtype=np.float64)
    grid = np.geomspace(lo, hi, n)
    return float(min(grid, key=lambda T: _nll(z, y, T)))


def ece(top1_probs, correct, n_bins: int = 10) -> float:
    """Expected calibration error of the top-1 confidence (equal-width bins)."""
    conf = np.asarray(top1_probs, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    b = np.linspace(0, 1, n_bins + 1)
    e, N = 0.0, len(conf)
    for i in range(n_bins):
        m = (conf > b[i]) & (conf <= b[i + 1]) if i else (conf >= 0) & (conf <= b[1])
        if m.sum():
            e += m.sum() / N * abs(y[m].mean() - conf[m].mean())
    return e


def _top1_conf_correct(probs, records, gene, names):
    """(top-1 sigmoid confidence, is-top-1-in-truth-set) over records that have a call for ``gene``."""
    top1 = probs.argmax(1)
    conf, corr = [], []
    for i, r in enumerate(records):
        tc = r.get(f"{gene}_call")
        if not tc:
            continue
        tset = {c.strip() for c in str(tc).split(",") if c.strip()}
        conf.append(float(probs[i, top1[i]]))
        corr.append(names[top1[i]] in tset)
    return np.asarray(conf), np.asarray(corr, dtype=np.float64)


def calibrate_allele_temperatures(model, reference, val_records, genes=("v", "d", "j"),
                                  device: str = "cpu") -> tuple[dict, dict]:
    """Fit a per-gene allele temperature on held-out ``val_records``.

    Returns ``(temperatures, report)`` where ``temperatures`` = ``{gene: T}`` and ``report`` gives the
    fitted ``T``, ECE before/after, and top-1 accuracy (unchanged by scaling) per gene."""
    from .forward import run_model
    cfg = model.cfg
    rc = SimpleNamespace(max_seq_length=cfg.max_seq_length, batch_size=128)
    raw = run_model(model, [str(r["sequence"]) for r in val_records], rc, device=device)
    out = {k: np.concatenate([b[k] for b in raw]) for k in raw[0] if k.endswith("_allele")}
    temps, report = {}, {}
    for g in genes:
        key = f"{g}_allele"
        if key not in out:
            continue
        names = list(reference.gene(g.upper()).names)
        conf, corr = _top1_conf_correct(out[key], val_records, g, names)
        if len(conf) < 50:                          # too few labelled samples to fit reliably
            continue
        T = fit_temperature(conf, corr)
        temps[g] = T
        report[g] = {"T": round(T, 4), "ece_before": round(ece(conf, corr), 4),
                     "ece_after": round(ece(apply_temperature(conf, T), corr), 4),
                     "acc": round(float(corr.mean()), 4), "n": int(len(conf))}
    return temps, report
