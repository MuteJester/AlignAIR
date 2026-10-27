"""Task-driven learning diagnostics over prepared model batches."""
from __future__ import annotations

from operator import index
import math

import torch

from ..core.config import AlignAIRConfig
from ..errors import ConfigurationError, error_context
from ..tasks import TrainingTasks, build_tasks
from .guards import NonFiniteLossError
from .setup import build_training_engine


def overfit_one_batch(model, batch_in, targets, *, steps=300, lr=1e-3,
                      genes=None, tasks: TrainingTasks | None = None, weights=None,
                      grad_clip=None) -> dict:
    """Update a model repeatedly on one prepared batch and report its final task metrics.

    Default tasks follow the model configuration, including all configured genes. Explicit tasks
    can define other losses and metrics; ``genes`` only selects a subset of the built-in tasks.
    History contains pre-update training losses; final loss and metrics use evaluation mode after
    the last update. The model and supplied weights are modified in place; the model ends in eval
    mode on success. Inputs/targets must already be on the model's device. This is a memorization
    diagnostic, not held-out evidence; the caller owns batch diversity and experiment seeding.
    """
    with error_context(stage="capacity_proof"):
        try:
            if isinstance(steps, bool) or index(steps) <= 0:
                raise ValueError
            steps = index(steps)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("steps must be a positive integer", context={"field": "steps"}) from exc
        if genes is not None:
            if tasks is not None:
                raise ConfigurationError("select genes through the supplied tasks, or use genes with default tasks")
            if not isinstance(getattr(model, "cfg", None), AlignAIRConfig):
                raise ConfigurationError("gene selection requires an AlignAIRConfig", context={"field": "genes"})
            tasks = build_tasks(model.cfg, genes=genes)
        if tasks is not None and not callable(getattr(tasks, "metrics", None)):
            raise ConfigurationError("capacity-proof tasks must supply metrics", context={"field": "tasks"})
        engine = build_training_engine(model, tasks=tasks, weights=weights, lr=lr, grad_clip=grad_clip)
        history = []
        for step in range(1, steps + 1):
            result = engine.step((batch_in, targets), step=step)
            history.append({"total": result.total, **result.parts})
        model.eval()
        with torch.no_grad(), error_context(stage="capacity_evaluation", step=steps):
            outputs = model(batch_in)
            total, parts = engine.tasks.loss(outputs, targets, engine.weights)
            final = float(total)
            part_values = {name: float(value) for name, value in parts.items()}
            if not all(math.isfinite(value) for value in (final, *part_values.values())):
                raise NonFiniteLossError("non-finite capacity-proof evaluation loss",
                                         context={"total": final, "parts": part_values})
            metrics = engine.tasks.metrics(outputs, targets)
        return {"initial_loss": history[0]["total"], "final_loss": final, "history": history, **metrics}
