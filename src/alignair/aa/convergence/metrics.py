"""Convergence-study metrics: macro accuracy, allele-frequency bins, softmax NLL, Brier."""
from __future__ import annotations
import numpy as np

def macro_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    accs = [(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]
    return float(np.mean(accs)) if accs else 0.0

def per_frequency_bins(y_true, y_pred, freq, edges) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); freq = np.asarray(freq)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == edges[-1]
        m = (freq >= lo) & (freq <= hi if last else freq < hi)
        label = f"[{lo},{hi}]" if last else f"[{lo},{hi})"
        out[label] = {"n": int(m.sum()),
                      "acc": float((y_pred[m] == y_true[m]).mean()) if m.any() else None}
    return out

def softmax_nll(probs, y_true) -> float:
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    p = probs[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))

def brier_score(probs, y_true) -> float:
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    oh = np.zeros_like(probs); oh[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))
