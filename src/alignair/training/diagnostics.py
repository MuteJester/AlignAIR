"""Training diagnostics: textbook + SOTA network-health analytics.

Instruments a training step to expose *why* and *how well* the network is learning: gradient health
(exploding/vanishing via norms + the update:weight ratio), weight health (norms + stable rank ->
representation collapse), activation health (dead / saturated units via forward hooks), Kendall
task-weighting, per-task held-out metrics, and numerical guards. Everything is cheap enough to run
periodically and is logged as JSONL for offline plotting.

Cheap metrics (grad/weight/logvar/numerical) run every few steps; deep metrics (activations +
held-out eval) run less often (they need an extra forward pass / hooks).
"""
from __future__ import annotations

import json
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
    sq = sum(float(t.detach().norm()) ** 2 for t in tensors if t is not None)
    return math.sqrt(sq)


def module_grad_weight(model, lr: float) -> dict:
    """Per top-level submodule: grad norm, weight norm, and the update:weight ratio (log10).

    ``update_ratio = log10(lr * ||grad|| / ||weight||)`` — Karpathy's health signal; healthy training
    hovers around -3 (updates ~0.1% of the weight magnitude). Much higher => unstable; much lower =>
    not learning.
    """
    out = {}
    for name, child in model.named_children():
        params = list(child.parameters())
        if not params:
            continue
        gnorm = _norm([p.grad for p in params])
        wnorm = _norm(params)
        ratio = math.log10(lr * gnorm / wnorm) if (wnorm > 0 and gnorm > 0) else float("nan")
        out[name] = {"grad_norm": gnorm, "weight_norm": wnorm, "update_ratio": ratio}
    return out


# --------------------------------------------------------------------------- weight / rank health
def stable_rank(w: torch.Tensor) -> float:
    """||W||_F^2 / ||W||_2^2 — a smooth rank proxy (1 = rank-1 collapse, up to min(dims))."""
    w = w.detach().reshape(w.shape[0], -1).float()
    fro_sq = float((w ** 2).sum())
    spec = float(torch.linalg.matrix_norm(w, ord=2))
    return fro_sq / (spec ** 2) if spec > 0 else 0.0


def weight_rank_stats(model, max_matrices: int = 24) -> dict:
    """Per named >=2D weight: norm, mean|w|, std, stable_rank (bounds cost with max_matrices)."""
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
def activation_stats(model, batch: dict, module_names=None, eps: float = 1e-6,
                     sat: float = 0.99) -> dict:
    """Forward-hook the named modules; report per-module output mean/std, dead-unit fraction
    (a channel whose max|activation| < eps across the batch), and saturation fraction
    (|activation| > sat — meaningful for tanh/sigmoid). One forward pass, hooks removed after."""
    if module_names is None:
        module_names = _default_hook_targets(model)
    named = dict(model.named_modules())
    stats, handles = {}, []

    def make_hook(nm):
        def hook(_m, _inp, out):
            t = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
            flat = t.reshape(-1, t.shape[-1]).float()                 # (samples, channels)
            dead = (flat.abs().amax(dim=0) < eps).float().mean()      # channel dead across batch
            mx = float(t.abs().max())
            bounded = mx <= 1.05                                      # tanh/sigmoid range
            # saturation is only meaningful for bounded activations; for unbounded layers report 0
            stats[nm] = {"mean": float(t.mean()), "std": float(t.std()), "dead_frac": float(dead),
                         "max_abs": mx, "bounded": bounded,
                         "sat_frac": float((t.abs() > sat).float().mean()) if bounded else 0.0}
        return hook

    for nm in module_names:
        if nm in named:
            handles.append(named[nm].register_forward_hook(make_hook(nm)))
    was_training = model.training
    model.eval()                               # eval-mode + no_grad => zero side effects on the model
    try:
        with torch.no_grad():
            model(batch)
    finally:                                   # ALWAYS restore mode + remove hooks (pure X-ray)
        model.train(was_training)
        for h in handles:
            h.remove()
    return stats


def _default_hook_targets(model) -> list:
    """A representative slice: embedding, orientation head, one post-tanh conv-bn per tower (bounded
    -> real tanh saturation), each tower's projection + the output heads (unbounded -> dead/explode)."""
    targets = ["embedding", "orientation_head"]
    for nm, _ in model.named_modules():
        if nm.endswith("conv_layers.0") or nm.endswith(".proj") or nm.endswith("_head") \
                or nm.startswith("cls_head") or nm.startswith("seg_heads"):
            targets.append(nm)
    return targets


# --------------------------------------------------------------------------- task weighting
def logvar_stats(logvars) -> dict:
    """Kendall log-var + effective weight exp(-s) per task (how the model balances the objectives)."""
    return {k: {"log_var": float(uw.log_var.detach()), "weight": float(torch.exp(-uw.log_var.detach()))}
            for k, uw in logvars.items()}


# --------------------------------------------------------------------------- numerical guards
def numerical_flags(loss, model) -> dict:
    bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.parameters())
    bad_w = any(not torch.isfinite(p).all() for p in model.parameters())
    return {"nonfinite_loss": not math.isfinite(float(loss)),
            "nonfinite_grad": bool(bad_grad), "nonfinite_weight": bool(bad_w)}


# --------------------------------------------------------------------------- per-task held-out eval
def eval_metrics(out: dict, targets: dict, cfg) -> dict:
    """Per-task quality on a batch: allele top-1-in-set accuracy, segmentation MAE, orientation
    accuracy, mutation/indel MAE, productivity accuracy."""
    genes = ["v", "j"] + (["d"] if cfg.has_d else [])
    m = {}
    for g in genes:
        pred = out[f"{g}_allele"]
        tgt = targets[f"{g}_allele"]
        top1 = pred.argmax(-1)
        in_set = tgt.gather(1, top1.unsqueeze(1)).squeeze(1)          # 1 if argmax is a true allele
        m[f"{g}_allele_top1"] = float((in_set > 0.5).float().mean())
        mae = 0.0
        for b in ("start", "end"):
            mae += float((out[f"{g}_{b}"].squeeze(-1) - targets[f"{g}_{b}"].squeeze(-1)).abs().mean())
        m[f"{g}_seg_mae"] = mae / 2
    if "orientation_logits" in out and "orientation" in targets:
        m["orientation_acc"] = float((out["orientation_logits"].argmax(-1)
                                      == targets["orientation"]).float().mean())
    m["mutation_mae"] = float((out["mutation_rate"] - targets["mutation_rate"]).abs().mean())
    m["indel_mae"] = float((out["indel_count"] - targets["indel_count"]).abs().mean())
    m["productive_acc"] = float(((out["productive"] > 0.5) == (targets["productive"] > 0.5)).float().mean())
    return m


# --------------------------------------------------------------------------- red-flag detection
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
        if a.get("sat_frac", 0) > 0.7:                 # bounded activations only (see activation_stats)
            flags.append(f"SATURATED:{name}({a['sat_frac']:.0%})")
        if a.get("max_abs", 0) > 100:
            flags.append(f"EXPLODING_ACT:{name}({a['max_abs']:.0f})")
    for name, w in (metrics.get("weights") or {}).items():
        if w.get("stable_rank", 1e9) < 2.0:
            flags.append(f"RANK_COLLAPSE:{name}({w['stable_rank']:.1f})")
    return flags


class TrainingMonitor:
    """Orchestrates the diagnostics: cheap metrics per call, deep metrics periodically, JSONL log,
    and a compact console health line with red flags."""

    def __init__(self, lr: float, log_path: str | None = None, deep_every: int = 500):
        self.lr = lr
        self.deep_every = deep_every
        self.fh = open(log_path, "a") if log_path else None

    def observe(self, model, logvars, loss, parts, step, eval_batch=None, cfg=None) -> dict:
        rec = {"step": step, "loss": float(loss),
               "parts": {k: float(v) for k, v in parts.items()},
               "grad_global": global_grad_norm(model.parameters()),
               "modules": module_grad_weight(model, self.lr),
               "logvars": logvar_stats(logvars),
               "numerical": numerical_flags(loss, model)}
        if step % self.deep_every == 0:
            rec["weights"] = weight_rank_stats(model)
            if eval_batch is not None and cfg is not None:
                rec["activations"] = activation_stats(model, eval_batch["input"])
                with torch.no_grad():
                    out = model(eval_batch["input"])
                rec["eval"] = eval_metrics(out, eval_batch["targets"], cfg)
        rec["flags"] = red_flags(rec)
        if self.fh:
            self.fh.write(json.dumps(rec) + "\n"); self.fh.flush()
        return rec

    @staticmethod
    def summary_line(rec: dict) -> str:
        parts = [f"g={rec['grad_global']:.1f}"]
        if "eval" in rec:
            ev = rec["eval"]
            parts.append("acc[" + " ".join(f"{g}={ev.get(f'{g}_allele_top1', 0):.2f}"
                                            for g in ("v", "d", "j") if f"{g}_allele_top1" in ev) + "]")
            if "orientation_acc" in ev:
                parts.append(f"ori={ev['orientation_acc']:.2f}")
        if rec.get("flags"):
            parts.append("!! " + " ".join(rec["flags"]))
        return "  ".join(parts)

    def close(self):
        if self.fh:
            self.fh.close()
