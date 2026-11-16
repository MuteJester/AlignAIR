"""Assemble the training state (the model, its AdamW optimizer, its schedule and its objective) without selecting a
source or checkpoint."""
from __future__ import annotations

import math

import torch
from torch import nn

from ..errors import ConfigurationError, error_context
from ..tasks import TrainingTasks
from .engine import TrainingEngine
from .guards import validate_grad_clip
from .schedules import CosineSchedule


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


def build_training_engine(model: nn.Module, *, tasks: TrainingTasks, device: str | torch.device | None = None,
                          lr: float = 3e-4, grad_clip=None, schedule: CosineSchedule | None = None) -> TrainingEngine:
    """Place the model, bind its parameters to a fresh AdamW, and create the engine that steps ``tasks`` (the model's
    objective, ``tasks.build_tasks``), under ``schedule`` (None: a constant rate).

    With device=None, retain the model's current single device (CPU for tensor-free models).
    Parameters are the model's, deduplicated by identity after placement; frozen parameters remain in the one group.
    Runners own training modes, checkpoint restoration, and update-budget policies.
    """
    with error_context(stage='training_setup'):
        if not isinstance(model, nn.Module):
            raise ConfigurationError('model must be a torch module', context={'field': 'model'})
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ConfigurationError('lr must be a positive finite number', context={'field': 'lr'})
        validate_grad_clip(grad_clip)
        if schedule is not None and not isinstance(schedule, CosineSchedule):
            raise ConfigurationError('schedule must be a CosineSchedule or None', context={'field': 'schedule'})
        if not callable(getattr(tasks, 'loss', None)):
            raise ConfigurationError('tasks must supply loss', context={'field': 'tasks'})
        target = _target_device(model, device)
        aliases = _parameter_aliases(model)
        model.to(target)
        if any(len({id(root.get_parameter(name)) for root, name in paths}) != 1 for paths in aliases):
            raise ConfigurationError('device placement split shared parameter aliases',
                hint='Place and tie the modules on the target device before assembling training state.')
        optimizer = torch.optim.AdamW(_owned_parameters(model), lr=lr)
        scheduler = schedule.build(optimizer) if schedule is not None else None
        return TrainingEngine(model, optimizer, tasks=tasks, grad_clip=grad_clip, scheduler=scheduler)
