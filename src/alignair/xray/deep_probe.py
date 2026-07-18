"""Expensive, anytime model-quality / generalization probes (run at checkpoints or post-training).

- **Neural collapse** (NC1/NC2/NC3) on a classification feature space + labels -> how cleanly the
  representation separates classes (the textbook 'has it truly converged' signal, beyond loss).
- **Weightwatcher power-law alpha** per weight matrix -> data-free generalization proxy (well-trained
  layers sit ~[2,4]; heavy tails / large alpha flag under-fit or over-parameterised layers).
- **Hessian sharpness** (top eigenvalue via power iteration, trace via Hutchinson) -> flatness of the
  minimum (flat => better generalization).
- **Classification margins** -> confidence/robustness of the decision boundary.

Model-agnostic: functions take tensors or closures. Read-only (autograd only; never touch ``.grad``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def neural_collapse(features: torch.Tensor, labels: torch.Tensor) -> dict:
    """NC metrics on penultimate ``features`` [N,D] with class ``labels`` [N] (for multi-label heads,
    pass the PRIMARY allele id). NC1 = within/between variance (↓ = collapse); NC2 = class-mean
    equinorm CV (↓); NC3 = deviation of class-mean cosines from the simplex-ETF value -1/(C-1) (↓)."""
    features, labels = features.float(), labels.long()
    classes = labels.unique()
    mu_g = features.mean(0)
    mus, within, n = [], 0.0, 0
    for c in classes:
        fc = features[labels == c]
        mu_c = fc.mean(0)
        mus.append(mu_c)
        within = within + float(((fc - mu_c) ** 2).sum())
        n += fc.shape[0]
    mus = torch.stack(mus)
    between = float(((mus - mu_g) ** 2).sum()) / max(1, len(classes))
    norms = (mus - mu_g).norm(dim=1)
    cm = F.normalize(mus - mu_g, dim=1)
    c = len(classes)
    cos = cm @ cm.t()
    off = cos[~torch.eye(c, dtype=torch.bool)] if c > 1 else torch.zeros(1)
    target = -1.0 / (c - 1) if c > 1 else 0.0
    return {"nc1_within_between": within / n / (between + 1e-9),
            "nc2_equinorm_cv": float(norms.std() / (norms.mean() + 1e-9)),
            "nc3_etf_deviation": float((off - target).abs().mean()), "n_classes": int(c)}


def weightwatcher_alpha(model, min_eigs: int = 10, max_matrices: int = 32) -> dict:
    """Per weight matrix: power-law exponent alpha of the eigenvalue tail of W Wᵀ (Hill estimator).
    Heavy-tailed self-regularization: well-trained layers ~ alpha in [2,4]."""
    out = {}
    for name, p in model.named_parameters():
        if p.dim() < 2 or "weight" not in name:
            continue
        sv = torch.linalg.svdvals(p.detach().reshape(p.shape[0], -1).float())
        ev = (sv ** 2)[sv > 1e-12].sort(descending=True).values
        if ev.numel() < min_eigs:
            continue
        k = max(min_eigs, ev.numel() // 2)
        tail = ev[:k]
        out[name] = float(1.0 + k / torch.log(tail / tail[-1]).sum())
        if len(out) >= max_matrices:
            break
    return out


def hessian_top_eigenvalue(loss_closure, params, iters: int = 20) -> float:
    """Top Hessian eigenvalue (curvature / sharpness) via power iteration on Hessian-vector products.
    ``loss_closure()`` must return the loss on a FIXED batch each call."""
    params = list(params)
    v = [torch.randn_like(p) for p in params]
    eig = 0.0
    for _ in range(iters):
        norm = torch.sqrt(sum((x ** 2).sum() for x in v))
        v = [x / (norm + 1e-12) for x in v]
        grads = torch.autograd.grad(loss_closure(), params, create_graph=True)
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=False)
        eig = float(sum((a * b).sum() for a, b in zip(hv, v)))
        v = [h.detach() for h in hv]
    return eig


def hessian_trace(loss_closure, params, n: int = 10) -> float:
    """Hessian trace (total curvature) via Hutchinson's estimator with Rademacher probes."""
    params = list(params)
    total = 0.0
    for _ in range(n):
        v = [torch.randint(0, 2, p.shape, device=p.device).float() * 2 - 1 for p in params]
        grads = torch.autograd.grad(loss_closure(), params, create_graph=True)
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=False)
        total += float(sum((a * b).sum() for a, b in zip(hv, v)))
    return total / n


def classification_margin(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Margin = true-class logit - max other-class logit, per sample. Larger/positive => robust."""
    labels = labels.long().view(-1, 1)
    true = logits.gather(1, labels).squeeze(1)
    masked = logits.clone().scatter_(1, labels, float("-inf"))
    margin = true - masked.max(1).values
    return {"mean": float(margin.mean()), "p10": float(margin.quantile(0.1)),
            "frac_negative": float((margin < 0).float().mean())}
