"""Atomic matcher checkpoints with data-only decoding and shared strict restoration.

These artifacts bind model configuration, optimizer state and completed step. They do not restore
source-stream position or process RNG state and are not exact replay checkpoints.
"""
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from operator import index
import pickle

import torch

from ..core.runtime import restore_model_state
from ..errors import ConfigurationError, ModelFormatError, OperationIOError, TrainingError, error_context
from ..io.atomic import atomic_write


def _step(value):
    try:
        if isinstance(value, bool) or index(value) < 0:
            raise ValueError
        return index(value)
    except (TypeError, ValueError) as exc:
        raise ModelFormatError("matcher checkpoint step must be a nonnegative integer") from exc


def save_checkpoint(path, cfg, model, opt, step):
    step = _step(step)
    cfg_d = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    state = {"config": cfg_d, "model": model.state_dict(),
             "opt": opt.state_dict(), "step": step}
    with error_context(stage="matcher_checkpoint_write", path=str(path)):
        try:
            atomic_write(path, lambda temporary: torch.save(state, temporary), stage="matcher_checkpoint_write")
        except OSError as exc:
            raise OperationIOError("cannot publish matcher checkpoint") from exc


def read_checkpoint(path, *, device="cpu") -> dict:
    """Decode tensor/builtin data once; never execute unrestricted pickle payloads."""
    with error_context(stage="matcher_checkpoint_read", path=str(path)):
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except OSError as exc:
            raise OperationIOError("cannot read matcher checkpoint") from exc
        except (pickle.UnpicklingError, EOFError, RuntimeError, ValueError) as exc:
            raise ModelFormatError("cannot decode matcher checkpoint") from exc
        if (not isinstance(state, Mapping) or not isinstance(state.get("model"), Mapping)
                or not isinstance(state.get("config"), Mapping)):
            raise ModelFormatError("matcher checkpoint requires model and config mappings")
        return {**state, "step": _step(state.get("step", 0))}


def load_checkpoint(path, model, opt=None, device="cpu"):
    """Strictly restore weights and optional optimizer; return the saved step.

    Resume requires the same configuration and an optimizer state. Weight-only warm starts permit
    configuration changes when all weight keys/shapes remain compatible. A failed optimizer restore
    is not a transactional rollback of model weights.
    """
    state = read_checkpoint(path, device=device)
    with error_context(stage="matcher_checkpoint_restore", path=str(path)):
        if opt is not None:
            if state["config"] != asdict(model.cfg):
                raise ConfigurationError("matcher resume configuration differs from the checkpoint")
            if not isinstance(state.get("opt"), Mapping):
                raise ModelFormatError("matcher resume requires saved optimizer state")
        restore_model_state(model, state["model"], source=path)
        if opt is not None:
            try:
                opt.load_state_dict(state["opt"])
            except (ValueError, KeyError, TypeError) as exc:
                raise TrainingError("cannot restore matcher optimizer state") from exc
    return state["step"]
