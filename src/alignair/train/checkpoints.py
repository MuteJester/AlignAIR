"""Atomic checkpoint publication, rotation, and explicit artifact contracts."""
from __future__ import annotations

import dataclasses
import glob
import logging
from operator import index
import os
import re
import shutil

from ..errors import ConfigurationError, error_context
from ..io.atomic import atomic_write
from .random_state import capture_rng_state


def save_checkpoint(path, model, step, opt=None, *, dataconfigs=None, train_args=None, label_schema=None,
                    scheduler=None):
    """Self-contained .alignair checkpoint, retaining embedded reference and trusted training state."""
    from .. import model_file as mf

    arguments = dict(train_args or {})
    training = {"steps": step, "batch_size": arguments.get("batch_size", 0), "lr": arguments.get("lr"),
                "progresses": arguments.get("progresses"), "heavy_shm": arguments.get("heavy_shm"),
                "short_boost": arguments.get("short_boost"), "seed": arguments.get("seed"), "train_args": arguments}
    rng = capture_rng_state()
    atomic_write(path, lambda temporary: mf.save_model(
        temporary, model, dataconfigs=dataconfigs or [], training=training,
        optimizer=opt, rng=rng, label_schema=label_schema, scheduler=scheduler), stage="training_checkpoint")


@dataclasses.dataclass(frozen=True)
class ResumeState:
    model_state: dict
    optimizer_state: dict | None
    step: int
    rng: dict
    config: object = None
    label_schema: dict | None = None
    scheduler_state: dict | None = None
    schedule: dict | None = None
    initial_lr: float | None = None


def validate_continuation(state, config, label_schema, *, schedule=None, lr=None):
    """Reject incompatible continuation before mutating model, optimizer or process RNG state."""
    if (state.config is not None and state.config != config
            or state.label_schema is not None and state.label_schema != label_schema):
        raise ConfigurationError('resume configuration or label identity differs from the requested model setup',
                                 hint='Use init_from for a weights-only warm start when changing model, label or loss settings.')
    if state.schedule != schedule or schedule is not None and (state.scheduler_state is None or state.initial_lr != lr):
        raise ConfigurationError('resume learning-rate schedule differs or its continuation state is missing')
    if state.optimizer_state is None:
        raise ConfigurationError('checkpoint has no optimizer continuation state; use init_from for a warm start')


def load_training_checkpoint(path, *, device="cpu") -> ResumeState:
    """Decode a self-contained .alignair checkpoint without allocating a model; restoration validates weight
    compatibility. The trusted resume reader decodes its pickle training sections; it does not apply the recorded
    process RNG state. An AlignAIR 3 checkpoint (a .pt) is refused with the version that reads it."""
    from .. import model_file as mf

    with error_context(stage="training_checkpoint_load", path=str(path)):
        if not mf.container.is_alignair_file(path):
            raise mf.v3_checkpoint_error(path)
        state = mf.read_training_state(path, device=device)
        return ResumeState(state.model_state, state.optimizer_state,
                           state.step, state.rng, state.config, state.metadata.get('model', {}).get('label_schema'),
                           state.scheduler_state,
                           state.train_args.get('schedule'), state.train_args.get('lr'))


def _ext(path) -> str:
    path = os.fspath(path)
    return next((extension for extension in (".alignair", ".pt") if path.endswith(extension)), ".alignair")


def _stem(path) -> str:
    path = os.fspath(path)
    extension = _ext(path)
    return path[:-len(extension)] if path.endswith(extension) else path


def _step_of(path) -> int:
    match = re.search(r"\.step(\d+)\.(?:alignair|pt)$", os.fspath(path))
    return int(match.group(1)) if match else -1


def save_rotating(base_path, model, step, opt, keep=3, *, dataconfigs=None, train_args=None, label_schema=None,
                  scheduler=None):
    """Keep numbered checkpoints and atomically refresh the convenient base-path copy."""
    try:
        if isinstance(keep, bool) or index(keep) <= 0:
            raise ValueError
        keep = index(keep)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError("checkpoint retention must be a positive integer") from exc
    stem, extension = _stem(base_path), _ext(base_path)
    path = f"{stem}.step{step}{extension}"
    save_checkpoint(path, model, step, opt, dataconfigs=dataconfigs, train_args=train_args,
                    label_schema=label_schema, scheduler=scheduler)
    atomic_write(base_path, lambda temporary: shutil.copyfile(path, temporary), stage="training_checkpoint")
    candidates = [candidate for candidate in glob.glob(f"{glob.escape(stem)}.step*{extension}")
                  if _step_of(candidate) >= 0]
    for old in sorted(candidates, key=_step_of)[:-keep]:
        if old != path:
            try:
                os.remove(old)
            except OSError as exc:
                logging.getLogger(__name__).warning("Could not remove old checkpoint %s: %s", old, exc)
    return path


def latest_checkpoint(base_path):
    stem = glob.escape(_stem(base_path))
    files = [path for extension in (".alignair", ".pt") for path in glob.glob(f"{stem}.step*{extension}")
             if _step_of(path) >= 0]
    return max(files, key=_step_of) if files else os.fspath(base_path) if os.path.exists(base_path) else None
