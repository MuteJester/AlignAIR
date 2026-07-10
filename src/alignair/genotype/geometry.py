"""Allele prototype geometry + the asymmetric leakage prior.

Each gene branch's final classification head (`cls_head`) is a ``Linear(latent, n_alleles)``; its weight
rows are per-allele **prototype vectors** in the model's latent space. Empirically, sibling (same-gene)
prototypes cluster (cosine ~0.76 vs ~0.58 random on v2.final), so cosine is a real confusability signal.

``LeakageModel`` turns that into an **asymmetric** expected-leakage matrix ``L[source->sibling]`` — the
fraction of a source allele's reads that spuriously call a sibling — rising with prototype cosine and
scaled by the sibling's "attractiveness" (head bias + relative prototype norm). It is a residual PRIOR:
`residual_support` subtracts predicted leakage from ALL confidently-present neighbors; it never decides.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def allele_prototypes(model, gene: str):
    """(weights, biases) of the gene's classification head: prototypes ``(n_alleles, latent)`` + bias."""
    head = model.branches[gene].cls_head
    W = head.weight.detach().cpu().numpy().astype(np.float64)
    b = (head.bias.detach().cpu().numpy().astype(np.float64)
         if head.bias is not None else np.zeros(W.shape[0]))
    return W, b


def prototype_cosine(W: np.ndarray) -> np.ndarray:
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return Wn @ Wn.T


class LeakageModel:
    """Asymmetric ``L[source->sibling]`` = clip(alpha·(cos−cos0), 0, cap)·attract[sibling]."""

    def __init__(self, cosine: np.ndarray, attract: np.ndarray, *, alpha=1.0, cos0=0.5, cap=0.5):
        self.cosine, self.attract = cosine, attract
        self.alpha, self.cos0, self.cap = alpha, cos0, cap

    @classmethod
    def fit(cls, W: np.ndarray, *, biases=None, norms=None, homozygous_leakage: dict | None = None,
            alpha: float = 1.0, cos0: float = 0.5, cap: float = 0.5):
        """Build from prototypes. ``biases`` (head bias) + prototype ``norms`` set attractiveness;
        ``alpha``/``cos0``/``cap`` are the (calibratable) leakage knobs; ``homozygous_leakage`` =
        {source: {sibling: observed_fraction}} from homozygous repertoires refits ``alpha``/``cos0``."""
        C = prototype_cosine(W)
        n = W.shape[0]
        bias = np.zeros(n) if biases is None else np.asarray(biases, dtype=np.float64)
        nrm = np.linalg.norm(W, axis=1) if norms is None else np.asarray(norms, dtype=np.float64)
        attract = _sigmoid(bias) * (nrm / (nrm.mean() + 1e-9))
        m = cls(C, attract, alpha=alpha, cos0=cos0, cap=cap)
        if homozygous_leakage:
            m._calibrate(homozygous_leakage)
        return m

    def _calibrate(self, homozygous_leakage: dict) -> None:
        xs, ys = [], []
        for src, sibs in homozygous_leakage.items():
            for j, frac in sibs.items():
                xs.append(self.cosine[src, j])
                ys.append(frac / (self.attract[j] + 1e-9))   # de-scale by attractiveness -> base(cos)
        xs, ys = np.asarray(xs), np.asarray(ys)
        if len(xs) >= 2 and xs.std() > 1e-9:                 # LS fit base = alpha*(cos - cos0)
            slope, intercept = np.linalg.lstsq(np.vstack([xs, np.ones_like(xs)]).T, ys, rcond=None)[0]
            self.alpha = max(0.0, float(slope))
            self.cos0 = float(-intercept / slope) if slope > 1e-9 else self.cos0
        else:                                                # single point: cap leakage at what we saw
            self.cap = min(self.cap, float(max(ys.max(), 0.0)))

    def predict(self, source: int, sibling: int) -> float:
        base = self.alpha * (self.cosine[source, sibling] - self.cos0)
        return float(np.clip(base * self.attract[sibling], 0.0, self.cap))


def residual_support(usage: dict, leakage: LeakageModel, present=None) -> dict:
    """Subtract predicted leakage INTO each allele from every confidently-present neighbor. A PRIOR
    only — the output residual mass feeds Stage 4 pruning, it decides nothing here."""
    present = set(present) if present is not None else {k for k in usage if usage[k] > 0}
    out = {}
    for a, mass in usage.items():
        leaked = sum(usage[s] * leakage.predict(s, a) for s in present if s != a)
        out[a] = max(0.0, mass - leaked)
    return out
