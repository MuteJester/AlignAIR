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
    """Evaluate without gradients, restoring every module's mode exactly on exit."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.set_grad_enabled(False):
            yield
    finally:
        for module, training in modes:
            module.training = training


def resolve_device(device: str | None = "auto") -> str:
    """Resolve a device request to a concrete backend. ``"auto"`` picks CUDA, else Apple MPS, else CPU.
    An explicit ``"cuda"``/``"mps"`` that is unavailable falls back to CPU (reported by the caller)."""
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    d = str(device)
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if d.startswith("mps"):
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return "cpu"
    return d
