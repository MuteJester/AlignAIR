"""Shared optimizer steps and support-weighted validation for alignment tasks."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from ..errors import ConfigurationError, TrainingError, error_context
from ..core.runtime import evaluation_mode
from ..tasks import MetricAccumulator, TrainingTasks, build_tasks
from .guards import NonFiniteLossError, check_finite_loss, validate_grad_clip


class StepResult(NamedTuple):
    total: float
    parts: dict[str, float]
    gradient_norm: float


class ValidationResult(NamedTuple):
    metrics: dict[str, float]
    score: float


def _parameters(optimizer):
    # Optimizers can include learned task weights and other parameters outside the model.
    parameters, seen = [], set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def task_weight_parameters(weights):
    """Include parameters registered on a ModuleDict itself as well as its child modules."""
    if isinstance(weights, torch.nn.ModuleDict):
        yield from weights.parameters()
    else:
        for weight in weights.values():
            yield from weight.parameters()


class TrainingEngine:
    """Own step mechanics; source sampling, schedules, and checkpoint policy belong to runners.

    The optimizer must include all trainable task weights. A scheduler, when supplied, advances
    once after a successful optimizer step. ``gradient_norm`` covers every optimized parameter.
    """

    def __init__(self, model, optimizer, *, tasks: TrainingTasks | None = None, weights=None,
                 grad_clip=None, scheduler=None):
        validate_grad_clip(grad_clip)
        self.model = model
        self.optimizer = optimizer
        self.tasks = build_tasks(model.cfg) if tasks is None else tasks
        self.weights = self.tasks.make_weights() if weights is None else weights
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        optimized = {id(parameter) for parameter in _parameters(optimizer)}
        if any(parameter.requires_grad and id(parameter) not in optimized
               for parameter in task_weight_parameters(self.weights)):
            raise ConfigurationError("the optimizer must include every trainable task weight",
                                     hint="Create task weights before the optimizer and include their parameters.")
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ConfigurationError("the scheduler must belong to the engine's optimizer")

    def step(self, batch, *, step: int = 0) -> StepResult:
        """Backpropagate one batch; reject non-finite loss/norm before optimizer or scheduler changes."""
        with error_context(stage="training_step", step=step):
            inputs, targets = batch
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            total, parts = self.tasks.loss(outputs, targets, self.weights)
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
                if isinstance(self.weights, torch.nn.ModuleDict):
                    names.update({id(parameter): f"tasks.{name}" for name, parameter in self.weights.named_parameters()})
                else:
                    for task, weight in self.weights.items():
                        names.update({id(parameter): f"tasks.{task}.{name}" for name, parameter in weight.named_parameters()})
                invalid = [names.get(id(parameter), f"optimizer[{i}]") for i, parameter in enumerate(parameters)
                           if not torch.isfinite(parameter.grad).all()]
                # Do not leave rejected gradients available for a caller to accidentally reuse.
                self.optimizer.zero_grad(set_to_none=True)
                raise NonFiniteLossError(
                    f"non-finite gradient norm at step {step} (loss was finite: {total_value:.3g})",
                    context={"stage": "training_gradient", "total": total_value, "parameters": invalid,
                             "reason": "nonfinite_gradient" if invalid else "gradient_norm_overflow"},
                    hint="Check task gradients or lower the learning rate.",
                )
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            clamp = getattr(self.model, "clamp_params", None)
            if clamp is not None:
                clamp()
            for weight in self.weights.values():
                constrain = getattr(weight, "apply_constraints", None)
                if constrain is not None:
                    constrain()
            if self.scheduler is not None:
                self.scheduler.step()
            return StepResult(total_value, part_values, gradient_norm)

    def validate(self, batches) -> ValidationResult:
        return validate(self.model, batches, getattr(self.model, "cfg", None), tasks=self.tasks)


def train_step(model, batch_in, targets, cfg, logvars, opt, *, step=0, grad_clip=None,
               tasks: TrainingTasks | None = None):
    """Historical stateless step interface, delegated to the shared engine."""
    engine = TrainingEngine(model, opt, tasks=build_tasks(cfg) if tasks is None else tasks,
                            weights=logvars, grad_clip=grad_clip)
    return engine.step((batch_in, targets), step=step)


def validate(model, val_batches, cfg, *, tasks: TrainingTasks | None = None) -> ValidationResult:
    """Aggregate supervised totals/counts and restore mode on both success and failure."""
    tasks = build_tasks(cfg) if tasks is None else tasks
    accumulator = MetricAccumulator()
    with evaluation_mode(model):
        for inputs, targets in val_batches:
            accumulator.update(tasks.metric_report(model(inputs), targets))
    metrics = accumulator.means()
    return ValidationResult(metrics, tasks.selection_score(metrics))
