"""Model-agnostic network-health metrics — textbook + SOTA analytics that work on ANY ``nn.Module``.

Pure, read-only functions (nothing here mutates the model): gradient health (norms + the update:weight
ratio), weight health (norms + stable rank), activation health (dead / saturated / exploding units via
forward hooks), uncertainty-weight introspection, numerical guards, and a red-flag summariser.
"""
from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------- gradient health
def global_grad_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().norm()) ** 2
    return math.sqrt(sq)


def _norm(tensors) -> float:
    return math.sqrt(sum(float(t.detach().norm()) ** 2 for t in tensors if t is not None))


def module_grad_weight(model, lr: float) -> dict:
    """Per top-level submodule: grad norm, weight norm, and the update:weight ratio (log10).

    ``update_ratio = log10(lr * ||grad|| / ||weight||)`` (Karpathy) — healthy training hovers ~-3
    (updates ~0.1% of the weight magnitude). Much higher => unstable; much lower => not learning.
    """
    out = {}
    for name, child in model.named_children():
        params = list(child.parameters())
        if not params:
            continue
        g, w = _norm([p.grad for p in params]), _norm(params)
        out[name] = {"grad_norm": g, "weight_norm": w,
                     "update_ratio": math.log10(lr * g / w) if (w > 0 and g > 0) else float("nan")}
    return out


# --------------------------------------------------------------------------- weight / rank health
def stable_rank(w: torch.Tensor) -> float:
    """||W||_F^2 / ||W||_2^2 — smooth rank proxy (1 = rank-1 collapse, up to min(dims))."""
    w = w.detach().reshape(w.shape[0], -1).float()
    fro_sq = float((w ** 2).sum())
    spec = float(torch.linalg.matrix_norm(w, ord=2))
    return fro_sq / (spec ** 2) if spec > 0 else 0.0


def weight_rank_stats(model, max_matrices: int = 24) -> dict:
    """Per named >=2D weight: norm, mean|w|, std, stable_rank (bounded by max_matrices for cost)."""
    out = {}
    for name, p in model.named_parameters():
        if p.dim() < 2 or "weight" not in name:
            continue
        w = p.detach()
        out[name] = {"norm": float(w.norm()), "mean_abs": float(w.abs().mean()),
                     "std": float(w.std()), "stable_rank": stable_rank(w)}
        if len(out) >= max_matrices:
            break
    return out


# --------------------------------------------------------------------------- activation health
def activation_stats(model, model_input, module_names=None, eps: float = 1e-6, sat: float = 0.99) -> dict:
    """Forward-hook the named modules; per-module output mean/std, dead-unit fraction, saturation
    fraction (bounded activations only), and max|activation|. **Read-only**: runs in ``eval()`` +
    ``no_grad`` and ``try/finally``-guarantees mode restore + hook removal, so the model is unchanged.
    """
    if module_names is None:
        module_names = default_hook_targets(model)
    named = dict(model.named_modules())
    stats, handles = {}, []

    def make_hook(nm):
        def hook(_m, _inp, out):
            t = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
            flat = t.reshape(-1, t.shape[-1]).float()
            dead = (flat.abs().amax(dim=0) < eps).float().mean()
            mx = float(t.abs().max())
            bounded = mx <= 1.05                              # tanh/sigmoid range
            stats[nm] = {"mean": float(t.mean()), "std": float(t.std()), "dead_frac": float(dead),
                         "max_abs": mx, "bounded": bounded,
                         "sat_frac": float((t.abs() > sat).float().mean()) if bounded else 0.0}
        return hook

    for nm in module_names:
        if nm in named:
            handles.append(named[nm].register_forward_hook(make_hook(nm)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(model_input)
    finally:
        model.train(was_training)
        for h in handles:
            h.remove()
    return stats


def capture_activations(model, model_input, module_names=None, max_samples: int = 512) -> dict:
    """Return {module_name: flattened activation matrix [N, D]} for geometry analysis. Read-only
    (eval + no_grad, hooks always removed). Rows are subsampled to ``max_samples`` to bound cost."""
    if module_names is None:
        module_names = default_hook_targets(model)
    named = dict(model.named_modules())
    caps, handles = {}, []

    def make_hook(nm):
        def hook(_m, _inp, out):
            t = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
            x = t.reshape(-1, t.shape[-1]).float()           # (samples, features)
            caps[nm] = x[:max_samples].cpu()
        return hook

    for nm in module_names:
        if nm in named:
            handles.append(named[nm].register_forward_hook(make_hook(nm)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(model_input)
    finally:
        model.train(was_training)
        for h in handles:
            h.remove()
    return caps


def default_hook_targets(model) -> list:
    """Heuristic representative slice for AlignAIR-style conv models: embedding, orientation head,
    one post-nonlinearity block per tower (bounded -> real saturation), tower projections + heads."""
    targets = [n for n in ("embedding", "orientation_head") if hasattr(model, n)]
    for nm, _ in model.named_modules():
        if nm.endswith("conv_layers.0") or nm.endswith(".proj") or nm.endswith("_head") \
                or nm.startswith("cls_head") or nm.startswith("seg_heads"):
            targets.append(nm)
    return targets


# --------------------------------------------------------------------------- uncertainty weights
def uncertainty_stats(named_uncertainty) -> dict:
    """{name: module-with-.log_var} -> {name: {log_var, weight=exp(-log_var)}}. For Kendall-style
    multi-task weighting: shows how the model balances / trusts each objective."""
    return {k: {"log_var": float(uw.log_var.detach()),
                "weight": float(torch.exp(-uw.log_var.detach()))}
            for k, uw in named_uncertainty.items()}


# --------------------------------------------------------------------------- numerical guards
def numerical_flags(loss, model) -> dict:
    bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
    bad_w = any(not torch.isfinite(p).all() for p in model.parameters())
    return {"nonfinite_loss": not math.isfinite(float(loss)),
            "nonfinite_grad": bool(bad_grad), "nonfinite_weight": bool(bad_w)}


# --------------------------------------------------------------------------- red flags
def red_flags(metrics: dict) -> list:
    flags = []
    num = metrics.get("numerical", {})
    if num.get("nonfinite_loss") or num.get("nonfinite_grad") or num.get("nonfinite_weight"):
        flags.append("NON_FINITE")
    if metrics.get("grad_global", 0) > 1e3:
        flags.append(f"EXPLODING_GRAD({metrics['grad_global']:.0f})")
    for name, r in (metrics.get("modules") or {}).items():
        ur = r.get("update_ratio")
        if ur is not None and math.isfinite(ur) and ur > -1.0:
            flags.append(f"HOT_UPDATES:{name}({ur:.1f})")
    for name, a in (metrics.get("activations") or {}).items():
        if a.get("dead_frac", 0) > 0.5:
            flags.append(f"DEAD:{name}({a['dead_frac']:.0%})")
        if a.get("sat_frac", 0) > 0.7:
            flags.append(f"SATURATED:{name}({a['sat_frac']:.0%})")
        if a.get("max_abs", 0) > 100:
            flags.append(f"EXPLODING_ACT:{name}({a['max_abs']:.0f})")
    for name, w in (metrics.get("weights") or {}).items():
        if w.get("stable_rank", 1e9) < 2.0:
            flags.append(f"RANK_COLLAPSE:{name}({w['stable_rank']:.1f})")
    return flags
