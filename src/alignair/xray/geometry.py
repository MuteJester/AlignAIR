"""Representation geometry — geometric/statistical properties of a feature space (activation matrix
``X`` of shape ``[N, D]``). Model-agnostic; operates on captured activations.

Answers "what does each feature space look like": how many dimensions it truly uses (effective /
participation / intrinsic rank), whether it has collapsed into a narrow cone (anisotropy / mean
cosine), how redundant its units are (collinearity), and how similar two layers' representations are
(linear CKA — for inter-layer relationships and cross-checkpoint drift).
"""
from __future__ import annotations

import torch

import torch.nn.functional as F


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(0, keepdim=True)


def _singular_values(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(_center(x.float()))


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of the normalized singular-value spectrum) — Roy & Vetterli. Smooth count of the
    dimensions the representation actually uses (1 = collapsed, up to min(N, D))."""
    s = _singular_values(x)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def participation_ratio(x: torch.Tensor) -> float:
    """(Σλ)²/Σλ² of the covariance spectrum — an alternative effective-dimensionality estimate."""
    lam = _singular_values(x) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if float(lam.sum()) > 0 else 0.0


def anisotropy(x: torch.Tensor) -> float:
    """Fraction of variance in the top principal direction (λ1/Σλ). High => collapsed toward a line."""
    lam = _singular_values(x) ** 2
    return float(lam.max() / lam.sum()) if float(lam.sum()) > 0 else 0.0


def mean_cosine(x: torch.Tensor, max_n: int = 256) -> float:
    """Average pairwise cosine between samples — the 'cone effect'. High => representations point the
    same way (little discriminative spread)."""
    x = x.float()[:max_n]
    xn = F.normalize(x, dim=1)
    c = xn @ xn.t()
    n = c.shape[0]
    return float((c.sum() - c.diagonal().sum()) / (n * (n - 1))) if n > 1 else 0.0


def feature_collinearity(x: torch.Tensor) -> float:
    """Mean |off-diagonal| of the feature correlation matrix — how redundant/collinear the units are
    (high => the layer's dimensions carry overlapping information; wasted capacity, possible
    interference)."""
    z = F.normalize(_center(x.float()), dim=0)          # unit-variance columns
    r = z.t() @ z
    d = r.shape[0]
    return float((r.abs().sum() - r.abs().diagonal().sum()) / (d * (d - 1))) if d > 1 else 0.0


def intrinsic_dimension(x: torch.Tensor, max_n: int = 512) -> float:
    """TwoNN intrinsic-dimension estimator (Facco et al.): the ID of the data manifold from the ratio
    of 2nd/1st nearest-neighbour distances. Often lower than the nominal dim; low ID tends to track
    better generalization."""
    x = x.float()[:max_n]
    n = x.shape[0]
    if n < 4:
        return float(x.shape[1])
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    r1, i1 = d.min(1)
    d2 = d.clone()
    d2[torch.arange(n), i1] = float("inf")
    r2, _ = d2.min(1)
    mu = (r2 / r1.clamp_min(1e-9)).clamp_min(1 + 1e-6)
    return float(n / torch.log(mu).sum())


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear Centered Kernel Alignment (Kornblith et al.) — representational similarity in [0,1] of
    two feature spaces on the SAME inputs. Use across layers (redundancy) and across checkpoints of
    the same layer (has this layer stopped changing => converged)."""
    xc, yc = _center(x.float()), _center(y.float())
    num = (yc.t() @ xc).pow(2).sum()
    den = (xc.t() @ xc).norm() * (yc.t() @ yc).norm()
    return float(num / den) if float(den) > 0 else 0.0


def layer_geometry(x: torch.Tensor) -> dict:
    """Full geometric fingerprint of one feature space."""
    return {"eff_rank": effective_rank(x), "participation_ratio": participation_ratio(x),
            "anisotropy": anisotropy(x), "mean_cosine": mean_cosine(x),
            "collinearity": feature_collinearity(x), "intrinsic_dim": intrinsic_dimension(x),
            "n": int(x.shape[0]), "dim": int(x.shape[1])}


def representation_geometry(activations: dict) -> dict:
    """{layer_name: activation[N,D]} -> {layer_name: geometry fingerprint}."""
    return {nm: layer_geometry(x) for nm, x in activations.items() if x.shape[0] >= 2 and x.shape[1] >= 2}


def cka_matrix(activations: dict) -> dict:
    """Pairwise linear CKA between all captured layers -> {'a|b': cka}. Shows which layers learn
    similar (redundant) representations."""
    names = list(activations)
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if activations[a].shape[0] == activations[b].shape[0]:
                out[f"{a}|{b}"] = linear_cka(activations[a], activations[b])
    return out
