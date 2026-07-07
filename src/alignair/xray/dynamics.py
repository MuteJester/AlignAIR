"""Training-dynamics analytics beyond the loss curve — read-only.

- Per-layer weight *velocity* (‖ΔW‖/‖W‖ between two snapshots) -> WHICH layers still change vs have
  converged (so "more training?" is answered per-layer, not just globally).
- Gradient SNR (‖E[g]‖/‖std[g]‖ across micro-batches) -> whether gradients are still signal or noise.
- **Multi-task gradient conflict** -> the cosine between each pair of task gradients on the SHARED
  trunk; a NEGATIVE cosine means the tasks pull the shared representation in opposing directions
  (one objective is 'combating' another). This is the direct instrument for interference/collinearity
  between feature spaces in a multi-task model.

All use ``torch.autograd.grad`` (does NOT touch ``.grad``) on a probe batch, so training is untouched.
"""
from __future__ import annotations

import torch


def weight_snapshot(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if p.dim() >= 2 and "weight" in n}


def weight_velocity(model, prev: dict) -> dict:
    """Per-layer relative change ‖W_now - W_prev‖ / ‖W_now‖ since the last snapshot (~0 => converged)."""
    out = {}
    for n, p in model.named_parameters():
        if n in prev:
            w = p.detach()
            wn = float(w.norm())
            out[n] = float((w - prev[n]).norm() / wn) if wn > 0 else 0.0
    return out


def _flat_grad(loss, params) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)])


def gradient_conflict(task_losses: dict, shared_params: list) -> dict:
    """task_losses: {name: loss_tensor} (from ONE forward, graphs retained). Returns per-task grad
    norm on the shared params and the pairwise cosine matrix; ``min_cosine`` is the worst conflict."""
    names = list(task_losses)
    gvec = {n: _flat_grad(task_losses[n], shared_params) for n in names}
    cos = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            denom = gvec[a].norm() * gvec[b].norm()
            cos[f"{a}|{b}"] = float((gvec[a] @ gvec[b]) / denom) if float(denom) > 0 else 0.0
    return {"grad_norm": {n: float(g.norm()) for n, g in gvec.items()}, "cosine": cos,
            "min_cosine": min(cos.values()) if cos else None}


def grad_snr(loss_fn, params, n_micro: int = 8) -> float:
    """Signal-to-noise of the gradient across ``n_micro`` fresh micro-batches: ‖mean(g)‖ / ‖std(g)‖.
    Falls with convergence (gradient becomes noise). ``loss_fn()`` returns a loss for a fresh batch."""
    gs = torch.stack([_flat_grad(loss_fn(), params) for _ in range(n_micro)])
    mean = gs.mean(0).norm()
    std = gs.std(0).norm()
    return float(mean / std) if float(std) > 0 else float("inf")
