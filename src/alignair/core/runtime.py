"""Shared execution contexts and strict state restoration for model instances."""
from contextlib import contextmanager

import torch

from ..errors import ModelIntegrityError


def restore_model_state(model, state_dict, *, source=None) -> None:
    """Apply decoded weights strictly to an existing instance, retaining parameter bindings and mode.

    PyTorch owns key/shape validation and copying; no replacement model is constructed. A failed
    load may have copied compatible tensors, so this is not a transactional rollback API.
    """
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ModelIntegrityError(
            f"cannot restore model weights: {exc}",
            context={"stage": "model_restoration", "path": source,
                     "architecture": getattr(getattr(model, "cfg", None), "architecture", None)},
            hint="Use weights and configuration from the same model artifact.") from exc


@contextmanager
def evaluation_mode(model):
    """Disable gradients and restore every module's prior mode, including on failure."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
