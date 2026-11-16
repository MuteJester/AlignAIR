"""Shared optimizer steps and support-weighted validation for alignment tasks."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from ..errors import ConfigurationError, TrainingError, error_context
from ..core.runtime import evaluation_mode
from ..tasks import MetricReport, TrainingTasks, build_tasks
from .guards import NonFiniteTrainingError, check_finite_loss, validate_grad_clip


class StepResult(NamedTuple):
    """One update's total loss, its terms and the gradient norm before clipping."""
    total: float
    parts: dict[str, float]
    gradient_norm: float


class ValidationResult(NamedTuple):
    """A validation pass's metrics and the score that selects the best checkpoint."""
    metrics: dict[str, float]
    score: float


def _parameters(optimizer):
    # An optimizer may hold parameters outside the model.
    parameters, seen = [], set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


class TrainingEngine:
    """Own step mechanics; source sampling, schedules, and checkpoint policy belong to runners.

    A scheduler, when supplied, advances once after a successful optimizer step. ``gradient_norm`` covers every
    optimized parameter.
    """

    def __init__(self, model, optimizer, *, tasks: TrainingTasks | None = None, grad_clip=None, scheduler=None):
        validate_grad_clip(grad_clip)
        self.model = model
        self.optimizer = optimizer
        self.tasks = build_tasks(model.cfg) if tasks is None else tasks
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ConfigurationError("the scheduler must belong to the engine's optimizer")

    def step(self, batch, *, step: int = 0) -> StepResult:
        """Backpropagate one batch; reject non-finite loss/norm before optimizer or scheduler changes."""
        with error_context(stage="training_step", step=step):
            inputs, targets = batch
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            total, parts = self.tasks.loss(outputs, targets)
            total_value = float(total.detach())
            part_values = {key: float(value.detach()) for key, value in parts.items()}
            check_finite_loss(step, total_value, part_values)
            if not total.requires_grad:
                raise TrainingError("the loss has no gradient graph", context={"stage": "training_gradient"})
            total.backward()
            parameters = [parameter for parameter in _parameters(self.optimizer) if parameter.grad is not None]
            if not parameters:
                raise TrainingError("the loss produced no gradients for any optimized parameter",
                                    context={"stage": "training_gradient"})
            norms = [parameter.grad.detach().norm(2) for parameter in parameters]
            norm = torch.stack([value.to(norms[0].device) for value in norms]).norm(2)
            gradient_norm = float(norm)
            if not math.isfinite(gradient_norm):
                names = {id(parameter): f"model.{name}" for name, parameter in self.model.named_parameters()}
                invalid = [names.get(id(parameter), f"optimizer[{i}]") for i, parameter in enumerate(parameters)
                           if not torch.isfinite(parameter.grad).all()]
                # Do not leave rejected gradients available for a caller to accidentally reuse.
                self.optimizer.zero_grad(set_to_none=True)
                raise NonFiniteTrainingError(
                    f"non-finite gradient norm at step {step} (loss was finite: {total_value:.3g})",
                    context={"stage": "training_gradient", "total": total_value, "parameters": invalid,
                             "reason": "nonfinite_gradient" if invalid else "gradient_norm_overflow"},
                    hint="Check task gradients or lower the learning rate.",
                )
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            return StepResult(total_value, part_values, gradient_norm)

    def validate(self, batches) -> ValidationResult:
        """The model's metrics and score on ``batches`` (``validate``)."""
        return validate(self.model, batches, tasks=self.tasks)


def validate(model, val_batches, *, tasks: TrainingTasks) -> ValidationResult:
    """Aggregate supervised totals/counts and restore mode on both success and failure."""
    report = MetricReport()
    with evaluation_mode(model):
        for inputs, targets in val_batches:
            report.merge(tasks.metric_report(model(inputs), targets))
    metrics = report.means()
    return ValidationResult(metrics, tasks.selection_score(metrics))
