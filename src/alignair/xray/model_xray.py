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

from . import network


class ModelXRay:
    def __init__(self, model, *, lr: float, log_path: Optional[str] = None, deep_every: int = 500,
                 uncertainty=None, task_eval: Optional[Callable] = None, hook_targets=None):
        """
        model         : the ``nn.Module`` under observation.
        lr            : learning rate (for the update:weight ratio).
        log_path      : JSONL file to append each record to (None = don't persist).
        deep_every    : cadence for expensive metrics (activations + weights + task_eval).
        uncertainty   : optional {name: module-with-.log_var} for Kendall task-weight introspection.
        task_eval     : optional ``callable(model, probe_input) -> {metric: float}`` for per-task quality.
        hook_targets  : optional explicit list of module names to hook for activations.
        """
        self.model = model
        self.lr = lr
        self.deep_every = deep_every
        self.uncertainty = uncertainty
        self.task_eval = task_eval
        self.hook_targets = hook_targets
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
            if probe_input is not None:
                rec["activations"] = network.activation_stats(self.model, probe_input, self.hook_targets)
                if self.task_eval is not None:
                    with torch.no_grad():
                        rec["eval"] = self.task_eval(self.model, probe_input)
        rec["flags"] = network.red_flags(rec)
        if self.fh:
            self.fh.write(json.dumps(rec) + "\n"); self.fh.flush()
        return rec

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
        if rec.get("flags"):
            parts.append("!! " + " ".join(rec["flags"]))
        return "  ".join(parts)

    def close(self):
        if self.fh:
            self.fh.close()
