"""Atomic checkpoint publication, rotation, and explicit artifact contracts."""
from __future__ import annotations

import dataclasses
import glob
import logging
from operator import index
import os
from pathlib import Path
import re
import shutil

import torch

from ..errors import ConfigurationError, error_context
from ..io.atomic import atomic_write
from .random_state import capture_rng_state


def save_checkpoint(path, cfg, model, logvars, step, opt=None, *, dataconfigs=None, train_args=None, label_schema=None,
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
        logvars=logvars, optimizer=opt, rng=rng, label_schema=label_schema, scheduler=scheduler), stage="training_checkpoint")


@dataclasses.dataclass(frozen=True)
class ResumeState:
    model_state: dict
    weight_state: dict | None
    optimizer_state: dict | None
    step: int
    rng: dict
    config: object = None
    label_schema: dict | None = None
    augmentation: dict | None = None
    scheduler_state: dict | None = None
    schedule: dict | None = None
    initial_lr: float | None = None


def validate_continuation(state, config, label_schema, *, augmentation=None, schedule=None, lr=None):
    """Reject incompatible continuation before mutating model, optimizer or process RNG state."""
    if (state.config is not None and state.config != config
            or state.label_schema is not None and state.label_schema != label_schema):
        raise ConfigurationError('resume configuration or label identity differs from the requested model setup')
    if state.augmentation != augmentation:
        raise ConfigurationError('resume augmentation policy differs from the requested training data',
                                 hint='Use init_from for a weights-only warm start when changing the augmentation policy.')
    if state.schedule != schedule or schedule is not None and (state.scheduler_state is None or state.initial_lr != lr):
        raise ConfigurationError('resume learning-rate schedule differs or its continuation state is missing')
    if state.optimizer_state is None:
        raise ConfigurationError('checkpoint has no optimizer continuation state; use init_from for a warm start')


def load_training_checkpoint(path, *, device="cpu", trust_pickle=False) -> ResumeState:
    """Normalize self-contained checkpoints and explicitly trusted historical NT .pt files."""
    from .. import model_file as mf
    from ..errors import ModelError

    with error_context(stage="training_checkpoint_load", path=str(path)):
        if mf.container.is_alignair_file(path):
            state = mf.load_training_state(path, device=device)
            return ResumeState(state.model.state_dict(), state.logvars_state, state.optimizer_state,
                               state.step, state.rng, state.config, state.metadata.get('model', {}).get('label_schema'),
                               state.train_args.get('augmentation'), state.scheduler_state,
                               state.train_args.get('schedule'), state.train_args.get('lr'))
        if not trust_pickle:
            raise ModelError("resuming a legacy .pt checkpoint runs torch.load (arbitrary-code pickle); "
                             "pass resume_trust_pickle=True only for a checkpoint you trust.")
        state = torch.load(path, map_location=device, weights_only=False)
        if 'logvars' not in state:
            return _research_training_state(state)
        return ResumeState(state["model"], state["logvars"], state.get("optimizer"),
                           int(state.get("step", 0)), state.get("rng") or {})


def validate_research_resume(checkpoint):
    """Historical crop checkpoints may continue only under their recorded target policy."""
    from collections.abc import Mapping
    from ..data.augmentation.partial import SAMPLE_POLICY_ID
    if checkpoint is not None and checkpoint.get('optimizer') is not None:
        provenance = checkpoint.get('provenance')
        if not isinstance(provenance, Mapping) or provenance.get('sample_policy') != SAMPLE_POLICY_ID:
            raise ConfigurationError('checkpoint crop-label policy differs from this trainer',
                                     hint='Use init_from for a weights-only warm start under the new label policy.')


def _research_training_state(state):
    """Explicitly trusted historical protein research state; absent label identity stays absent."""
    from ..core.config import AlignAIRConfig
    from ..data.augmentation import ProteinCropPolicy
    from .schedules import CosineSchedule
    validate_research_resume(state)
    try:
        config = dict(state['cfg'])
        config.setdefault('input_space', 'protein')
        config.setdefault('label_space', 'protein_classes')
        config = AlignAIRConfig(**config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError('research checkpoint has no usable model configuration',
                                 hint='Use an explicit weights-only warm start for incomplete historical snapshots.') from exc
    if state.get('optimizer') is None:
        return ResumeState(state['model'], None, None, int(state.get('step') or 0), {}, config)
    scheduler = state.get('scheduler')
    if not isinstance(scheduler, dict) or 'T_max' not in scheduler or 'eta_min' not in scheduler:
        raise ConfigurationError('crop continuation requires a saved cosine scheduler')
    hyperparams = state['provenance'].get('hyperparams', {})
    if not isinstance(hyperparams, dict):
        raise ConfigurationError('research checkpoint hyperparameters must be a mapping')
    augmentation = ProteinCropPolicy(hyperparams.get('p_crop', .4), hyperparams.get('p_sub', .4),
                                     hyperparams.get('p_il', .15)).to_dict()
    schedule = CosineSchedule(scheduler['T_max'], scheduler['eta_min']).to_dict()
    try:
        initial_lr = hyperparams.get('base_lr', state['optimizer']['param_groups'][0].get('initial_lr'))
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ConfigurationError('research checkpoint has no usable optimizer parameter groups') from exc
    return ResumeState(state['model'], None, state['optimizer'], int(state['step']), research_checkpoint_rng(state),
                       config, None, augmentation, scheduler, schedule, initial_lr)


def save_snapshot(path, model, cfg, step):
    """Weights/config/step only: a warm-start snapshot, without optimizer or RNG continuation."""
    config = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg
    atomic_write(path, lambda temporary: torch.save(
        {"model": model.state_dict(), "cfg": config, "step": step}, temporary), stage="training_checkpoint")


def save_research_checkpoint(path, *, model, cfg, step, optimizer, scheduler, lr, history, provenance):
    """Trusted Python research state, retaining the historical field schema and adding Python RNG."""
    rng = capture_rng_state()
    state = {"model": model.state_dict(), "cfg": cfg, "step": step,
             "optimizer": optimizer.state_dict() if optimizer is not None else None,
             "scheduler": scheduler.state_dict() if scheduler is not None else None,
             "rng_python": rng["python"], "rng_numpy": rng["numpy"], "rng_torch": rng["torch"],
             "rng_cuda": rng.get("cuda"), "lr": lr, "history": history, "provenance": provenance}
    atomic_write(path, lambda temporary: torch.save(state, temporary), stage="training_checkpoint")


def load_research_checkpoint(path) -> dict:
    """Load a trusted local .pt checkpoint; missing old-format state remains explicitly absent."""
    with error_context(stage="training_checkpoint_load", path=str(path)):
        state = torch.load(path, map_location="cpu", weights_only=False)
    return {key: state.get(key) for key in (
        "model", "cfg", "step", "optimizer", "scheduler", "rng_python", "rng_numpy", "rng_torch", "rng_cuda",
        "lr", "history", "provenance")}


def research_checkpoint_rng(state):
    """Translate flat research fields to the shared process-generator contract."""
    return {name: state.get(f"rng_{name}") for name in ("python", "numpy", "torch", "cuda")}


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


def save_rotating(base_path, cfg, model, logvars, step, opt, keep=3, *, dataconfigs=None, train_args=None, label_schema=None,
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
    save_checkpoint(path, cfg, model, logvars, step, opt, dataconfigs=dataconfigs, train_args=train_args,
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
