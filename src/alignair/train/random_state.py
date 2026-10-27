"""One process RNG-state contract for training and research checkpoints."""
from __future__ import annotations

import random

import numpy as np
import torch

from ..errors import TrainingError


def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _byte_tensor(state):
    value = torch.as_tensor(state, device="cpu")
    if value.ndim != 1 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError("RNG state must be a byte vector")
    if value.dtype != torch.uint8 and ((value < 0) | (value > 255)).any():
        raise ValueError("RNG state contains a value outside the byte range")
    return value.detach().to(dtype=torch.uint8)


def _apply(state):
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(_byte_tensor(state["torch"]))
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_byte_tensor(value) for value in state["cuda"]])


def restore_rng_state(state: dict) -> None:
    """Restore supplied generators, rolling back partial restoration on malformed state.

    CUDA state is retained in files but skipped on a CPU host. A source stream or a local NumPy
    Generator has its own position/state; restoring these process generators does not restore it.
    """
    if not state:
        return
    previous = capture_rng_state()
    try:
        _apply(state)
    except (RuntimeError, ValueError, TypeError) as exc:
        _apply(previous)
        raise TrainingError("checkpoint RNG state cannot be restored",
                            context={"stage": "training_resume_rng"}) from exc
