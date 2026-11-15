"""One process RNG-state contract for training and research checkpoints."""
from __future__ import annotations

import random
from collections.abc import Mapping

import numpy as np
import torch

from ..errors import TrainingError


def capture_rng_state(*, include_cuda: bool = True) -> dict:
    """Capture process generators, optionally excluding CUDA even on a GPU host."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if include_cuda and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def same_rng_state(left: dict, right: dict) -> bool:
    """Whether two CPU random states (``capture_rng_state(include_cuda=False)``) are equal, generator by generator."""
    return (left['python'] == right['python'] and torch.equal(left['torch'], right['torch'])
            and left['numpy'][0] == right['numpy'][0] and np.array_equal(left['numpy'][1], right['numpy'][1])
            and left['numpy'][2:] == right['numpy'][2:])


def _byte_tensor(state):
    value = torch.as_tensor(state, device="cpu")
    if value.ndim != 1 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError("RNG state must be a byte vector")
    if value.dtype != torch.uint8 and ((value < 0) | (value > 255)).any():
        raise ValueError("RNG state contains a value outside the byte range")
    return value.detach().to(dtype=torch.uint8)


def _apply(state, *, include_cuda):
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(_byte_tensor(state["torch"]))
    if include_cuda and state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_byte_tensor(value) for value in state["cuda"]])


def restore_rng_state(state: dict, *, include_cuda: bool = True, require_process_state: bool = False) -> None:
    """Restore supplied generators, rolling back partial restoration on malformed state.

    CUDA state is retained in files but skipped on a CPU host. A source stream or a local NumPy
    Generator has its own position/state; restoring these process generators does not restore it.
    Set include_cuda=False to exclude CUDA capture, restoration and rollback on any host.
    Sparse/None state is accepted by default; require_process_state requires non-null Python,
    NumPy and PyTorch CPU states before applying anything. CUDA remains independently optional.
    """
    if state is None and not require_process_state:
        return
    if not isinstance(state, Mapping):
        raise TrainingError("checkpoint RNG state must be a mapping",
                            context={"stage": "training_resume_rng", "field": "rng"})
    missing = [name for name in ("python", "numpy", "torch") if state.get(name) is None] if require_process_state else []
    if missing:
        raise TrainingError("checkpoint RNG state is incomplete",
                            context={"stage": "training_resume_rng", "fields": missing})
    if not state:
        return
    previous = capture_rng_state(include_cuda=include_cuda)
    try:
        _apply(state, include_cuda=include_cuda)
    except (RuntimeError, ValueError, TypeError, IndexError) as exc:
        _apply(previous, include_cuda=include_cuda)
        raise TrainingError("checkpoint RNG state cannot be restored",
                            context={"stage": "training_resume_rng"}) from exc
