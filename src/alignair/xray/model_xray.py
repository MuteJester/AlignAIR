"""``ModelXRay`` — a reusable, read-only training X-ray for any PyTorch model.

Attach it to a model and call :meth:`observe` each logging step; it emits a data-rich record
(gradient / weight / activation / rank health, uncertainty weights, optional per-task eval,
numerical guards + red flags) to JSONL so training can be *assessed on grounded attributes*, not
guessed from the loss curve. It never mutates the model (see ``network.activation_stats``).

Model-agnostic by design: the core health metrics work on any ``nn.Module``; anything model-specific
(per-task quality) is supplied via the ``task_eval`` callable.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

import torch

from . import deep_probe, dynamics, geometry, network


class ModelXRay:
    def __init__(self, model, *, lr: float, log_path: Optional[str] = None, deep_every: int = 500,
                 uncertainty=None, task_eval: Optional[Callable] = None,
                 task_losses: Optional[Callable] = None, shared_module: Optional[str] = None,
                 hook_targets=None):
        """
        model         : the ``nn.Module`` under observation.
        lr            : learning rate (for the update:weight ratio).
        log_path      : JSONL file to append each record to (None = don't persist).
        deep_every    : cadence for the deep pass (activations, weights, geometry, velocity, conflict).
        uncertainty   : optional {name: module-with-.log_var} for Kendall task-weight introspection.
        task_eval     : optional ``callable(model, probe_input) -> {metric: float}`` for per-task quality.
        task_losses   : optional ``callable(model, probe_input) -> {task: loss_tensor}`` (one forward,
                        graphs retained) enabling the multi-task gradient-conflict / interference metric.
        shared_module : name of the shared-trunk submodule whose params carry cross-task gradients
                        (gradient conflict + Hessian). Default: the model's first child.
        hook_targets  : optional explicit list of module names to hook for activations.
        """
        self.model = model
        self.lr = lr
        self.deep_every = deep_every
        self.uncertainty = uncertainty
        self.task_eval = task_eval
        self.task_losses = task_losses
        self.hook_targets = hook_targets
        self._prev_weights = None
        if shared_module and hasattr(model, shared_module):
            self._shared = list(getattr(model, shared_module).parameters())
        else:
            first = next(model.children(), None)
            self._shared = list(first.parameters()) if first is not None else list(model.parameters())
        self.fh = open(log_path, "a") if log_path else None

    def observe(self, step: int, loss, parts: Optional[dict] = None, probe_input=None) -> dict:
        """Cheap metrics every call; deep metrics (activations/weights/task_eval) every ``deep_every``.
        Reads the gradients currently on the model (call right after ``backward``/``step``, before the
        next ``zero_grad``). Returns the record (also written to the JSONL log)."""
        rec = {"step": step, "loss": float(loss),
               "grad_global": network.global_grad_norm(self.model.parameters()),
               "modules": network.module_grad_weight(self.model, self.lr),
               "numerical": network.numerical_flags(loss, self.model)}
        if parts is not None:
            rec["parts"] = {k: float(v) for k, v in parts.items()}
        if self.uncertainty is not None:
            rec["uncertainty"] = network.uncertainty_stats(self.uncertainty)
        if step % self.deep_every == 0:
            rec["weights"] = network.weight_rank_stats(self.model)
            if self._prev_weights is not None:                        # per-layer convergence velocity
                rec["velocity"] = dynamics.weight_velocity(self.model, self._prev_weights)
            self._prev_weights = dynamics.weight_snapshot(self.model)
            if probe_input is not None:
                rec["activations"] = network.activation_stats(self.model, probe_input, self.hook_targets)
                caps = network.capture_activations(self.model, probe_input, self.hook_targets)
                rec["geometry"] = geometry.representation_geometry(caps)      # per-layer feature geometry
                if self.task_eval is not None:
                    with torch.no_grad():
                        rec["eval"] = self.task_eval(self.model, probe_input)
                if self.task_losses is not None:
                    rec["interference"] = self._conflict(probe_input)        # cross-task gradient conflict
        rec["flags"] = network.red_flags(rec)
        if self.fh:
            self.fh.write(json.dumps(rec) + "\n"); self.fh.flush()
        return rec

    def _conflict(self, probe_input) -> dict:
        was = self.model.training
        self.model.eval()                          # BN frozen -> non-intrusive; grads still flow
        try:
            return dynamics.gradient_conflict(self.task_losses(self.model, probe_input), self._shared)
        finally:
            self.model.train(was)

    def deep_report(self, probe_input=None, *, loss_closure=None, nc_features=None, nc_labels=None,
                    margin_logits=None, margin_labels=None, hessian_iters: int = 15) -> dict:
        """Expensive quality / generalization probes — run at a checkpoint or after training. Composes
        whatever inputs are supplied (weightwatcher always; CKA if probe given; neural collapse /
        margins / Hessian sharpness if their inputs/closure are given). All read-only."""
        rep = {"weightwatcher_alpha": deep_probe.weightwatcher_alpha(self.model)}
        if probe_input is not None:
            caps = network.capture_activations(self.model, probe_input, self.hook_targets)
            rep["cka"] = geometry.cka_matrix(caps)
        if nc_features is not None and nc_labels is not None:
            rep["neural_collapse"] = deep_probe.neural_collapse(nc_features, nc_labels)
        if margin_logits is not None and margin_labels is not None:
            rep["margin"] = deep_probe.classification_margin(margin_logits, margin_labels)
        if loss_closure is not None:
            rep["hessian_top_eig"] = deep_probe.hessian_top_eigenvalue(loss_closure, self._shared, hessian_iters)
            rep["hessian_trace"] = deep_probe.hessian_trace(loss_closure, self._shared)
        return rep

    @staticmethod
    def summary_line(rec: dict) -> str:
        parts = [f"g={rec['grad_global']:.1f}"]
        ev = rec.get("eval")
        if ev:
            acc = " ".join(f"{g}={ev[f'{g}_allele_top1']:.2f}"
                           for g in ("v", "d", "j") if f"{g}_allele_top1" in ev)
            if acc:
                parts.append(f"acc[{acc}]")
            if "orientation_acc" in ev:
                parts.append(f"ori={ev['orientation_acc']:.2f}")
        if rec.get("interference", {}).get("min_cosine") is not None:
            parts.append(f"conflict={rec['interference']['min_cosine']:+.2f}")
        if rec.get("flags"):
            parts.append("!! " + " ".join(rec["flags"]))
        return "  ".join(parts)

    def close(self):
        if self.fh:
            self.fh.close()
