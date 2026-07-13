"""Training safety guards (P0-10): validate a training request before allocating the model, abort on
non-finite loss/gradients with a diagnostic, and bound gradient norms — so a run fails fast and loudly
on a bad config or a diverging optimizer instead of silently producing a broken model."""
from __future__ import annotations

import math


class TrainingConfigError(ValueError):
    """A training request is invalid (bad hyperparameters / config / reference). Raised *before* the
    full model is allocated so misuse fails fast with an actionable message."""


class NonFiniteLossError(RuntimeError):
    """Loss or a gradient became NaN/Inf — training is aborted with a per-task diagnostic rather than
    continuing to corrupt the weights."""


def validate_training_request(*, steps, batch_size, lr, max_seq_length, reference, progresses=(0.3,),
                              heavy_shm=0.0, short_boost=1, grad_clip=None) -> None:
    """Validate hyperparameters/config/reference up front. Raises :class:`TrainingConfigError` on the
    first problem. Call before building the model/optimizer so a typo doesn't waste a long allocation."""
    problems = []
    if not (isinstance(steps, int) and steps > 0):
        problems.append(f"steps must be a positive int (got {steps!r})")
    if not (isinstance(batch_size, int) and batch_size > 0):
        problems.append(f"batch_size must be a positive int (got {batch_size!r})")
    if not (isinstance(lr, (int, float)) and lr > 0 and math.isfinite(lr)):
        problems.append(f"lr must be a positive finite number (got {lr!r})")
    if not (isinstance(max_seq_length, int) and max_seq_length > 0):
        problems.append(f"max_seq_length must be a positive int (got {max_seq_length!r})")
    if not progresses or any(not (0.0 <= float(p) <= 1.0) for p in progresses):
        problems.append(f"progresses must be non-empty and each in [0, 1] (got {progresses!r})")
    if not (0.0 <= float(heavy_shm) <= 1.0):
        problems.append(f"heavy_shm must be in [0, 1] (got {heavy_shm!r})")
    if not (isinstance(short_boost, int) and short_boost >= 1):
        problems.append(f"short_boost must be an int >= 1 (got {short_boost!r})")
    if grad_clip is not None and not (isinstance(grad_clip, (int, float)) and grad_clip > 0):
        problems.append(f"grad_clip must be a positive number or None (got {grad_clip!r})")
    # reference must be non-empty for the genes the model will train (V and J are always required)
    try:
        n_v = len(reference.gene("V"))
        n_j = len(reference.gene("J"))
    except Exception as e:                              # noqa: BLE001 - any malformed reference
        problems.append(f"reference is unusable: {e}")
    else:
        if n_v == 0 or n_j == 0:
            problems.append(f"reference has empty V ({n_v}) or J ({n_j}) allele set")
    if problems:
        raise TrainingConfigError("invalid training request:\n  - " + "\n  - ".join(problems))


def check_finite_loss(step: int, total: float, parts: dict) -> None:
    """Abort with a per-task diagnostic if the loss is NaN/Inf (so weights aren't silently corrupted)."""
    if math.isfinite(float(total)):
        return
    diag = ", ".join(f"{k}={v:.3g}" for k, v in parts.items())
    raise NonFiniteLossError(f"non-finite loss {total!r} at step {step}; per-task: {diag}. "
                             f"Lower the learning rate, add gradient clipping (--grad-clip), or check "
                             f"the data/targets.")
