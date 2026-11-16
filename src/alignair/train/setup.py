"""Assemble shared training state without selecting a source, checkpoint, or study policy."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import torch
from torch import nn

from ..core.config import AlignAIRConfig
from ..errors import ConfigurationError, error_context
from ..tasks import TrainingTasks, build_tasks
from .engine import TrainingEngine
from .guards import validate_grad_clip


class OptimizerFactory(Protocol):
    """Create parameter groups from the supplied model parameters and learning rate."""
    def __call__(self, parameters: tuple[nn.Parameter, ...], *, lr: float) -> torch.optim.Optimizer: ...


def _target_device(model, device):
    if device is None:
        devices = {tensor.device for tensor in (*model.parameters(), *model.buffers())}
        if len(devices) > 1:
            raise ConfigurationError('specify a device for a model with tensors on multiple devices',
                                     context={'field': 'device'})
        return next(iter(devices), torch.device('cpu'))
    try:
        return torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigurationError('invalid training device', context={'field': 'device'}) from exc


def _parameter_aliases(model):
    owners = {}
    for module in model.modules():          # each module once, however many paths reach it
        for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            owners.setdefault(id(parameter), []).append((module, name))
    return tuple(paths for paths in owners.values() if len(paths) > 1)


def _owned_parameters(model):
    parameters = tuple(model.parameters())   # each parameter once, however many modules register it
    if not any(parameter.requires_grad for parameter in parameters):
        raise ConfigurationError('training requires at least one trainable model parameter')
    return parameters


def _validate_optimizer(optimizer, parameters):
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ConfigurationError('optimizer_factory must return a torch optimizer')
    expected = {id(parameter) for parameter in parameters}
    trainable = {id(parameter) for parameter in parameters if parameter.requires_grad}
    actual = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    if len(actual) != len(set(actual)) or not trainable <= set(actual) or not set(actual) <= expected:
        raise ConfigurationError('optimizer must own every trainable model parameter exactly once without foreign parameters',
                                 context={'field': 'optimizer_factory'})


def build_training_engine(model: nn.Module, *, tasks: TrainingTasks | None = None,
                          device: str | torch.device | None = None, lr: float = 3e-4, grad_clip=None,
                          optimizer_factory: OptimizerFactory = torch.optim.AdamW,
                          scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None) -> TrainingEngine:
    """Place modules, bind their parameters, and create one engine with fresh optimizer state.

    With device=None, retain the model's current single device (CPU for tensor-free models).
    Parameters are the model's, deduplicated by identity after placement; frozen parameters remain in default groups.
    Runners own training modes, checkpoint restoration, and update-budget/scheduler policies.
    """
    with error_context(stage='training_setup'):
        if not isinstance(model, nn.Module):
            raise ConfigurationError('model must be a torch module', context={'field': 'model'})
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ConfigurationError('lr must be a positive finite number', context={'field': 'lr'})
        validate_grad_clip(grad_clip)
        if not callable(optimizer_factory) or scheduler_factory is not None and not callable(scheduler_factory):
            raise ConfigurationError('optimizer and scheduler factories must be callable')
        target = _target_device(model, device)
        if tasks is None:
            if not isinstance(getattr(model, 'cfg', None), AlignAIRConfig):
                raise ConfigurationError('default tasks require an AlignAIRConfig; supply explicit tasks for other models',
                                         context={'field': 'tasks'})
            tasks = build_tasks(model.cfg)
        if not callable(getattr(tasks, 'loss', None)):
            raise ConfigurationError('tasks must supply loss', context={'field': 'tasks'})
        aliases = _parameter_aliases(model)
        model.to(target)
        if any(len({id(root.get_parameter(name)) for root, name in paths}) != 1 for paths in aliases):
            raise ConfigurationError('device placement split shared parameter aliases',
                hint='Place and tie the modules on the target device before assembling training state.')
        parameters = _owned_parameters(model)
        optimizer = optimizer_factory(parameters, lr=lr)
        _validate_optimizer(optimizer, parameters)
        scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None
        if scheduler_factory is not None and (isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                                               or not callable(getattr(scheduler, 'step', None))
                                               or getattr(scheduler, 'optimizer', None) is not optimizer):
            raise ConfigurationError('scheduler_factory must return a step scheduler bound to this optimizer',
                                     context={'field': 'scheduler_factory'})
        return TrainingEngine(model, optimizer, tasks=tasks, grad_clip=grad_clip, scheduler=scheduler)
