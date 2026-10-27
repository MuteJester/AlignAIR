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
import tempfile

import torch

from ..errors import ConfigurationError, error_context
from .random_state import capture_rng_state


def _atomic_write(path, writer):
    """Publish on the same filesystem; a failed writer leaves the previous artifact intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    try:
        with error_context(stage="training_checkpoint", path=str(path)):
            writer(temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, cfg, model, logvars, step, opt=None, *, dataconfigs=None, train_args=None, label_schema=None):
    """Self-contained .alignair checkpoint, retaining embedded reference and trusted training state."""
    from .. import model_file as mf

    arguments = dict(train_args or {})
    training = {"steps": step, "batch_size": arguments.get("batch_size", 0), "lr": arguments.get("lr"),
                "progresses": arguments.get("progresses"), "heavy_shm": arguments.get("heavy_shm"),
                "short_boost": arguments.get("short_boost"), "seed": arguments.get("seed"), "train_args": arguments}
    rng = capture_rng_state()
    _atomic_write(path, lambda temporary: mf.save_model(
        temporary, model, dataconfigs=dataconfigs or [], training=training,
        logvars=logvars, optimizer=opt, rng=rng, label_schema=label_schema))


@dataclasses.dataclass(frozen=True)
class ResumeState:
    model_state: dict
    weight_state: dict | None
    optimizer_state: dict | None
    step: int
    rng: dict
    config: object = None
    label_schema: dict | None = None


def load_training_checkpoint(path, *, device="cpu", trust_pickle=False) -> ResumeState:
    """Normalize self-contained checkpoints and explicitly trusted historical NT .pt files."""
    from .. import model_file as mf
    from ..errors import ModelError

    with error_context(stage="training_checkpoint_load", path=str(path)):
        if mf.container.is_alignair_file(path):
            state = mf.load_training_state(path, device=device)
            return ResumeState(state.model.state_dict(), state.logvars_state, state.optimizer_state,
                               state.step, state.rng, state.config, state.metadata.get('model', {}).get('label_schema'))
        if not trust_pickle:
            raise ModelError("resuming a legacy .pt checkpoint runs torch.load (arbitrary-code pickle); "
                             "pass resume_trust_pickle=True only for a checkpoint you trust.")
        state = torch.load(path, map_location=device, weights_only=False)
        return ResumeState(state["model"], state["logvars"], state.get("optimizer"),
                           int(state.get("step", 0)), state.get("rng") or {})


def save_snapshot(path, model, cfg, step):
    """Weights/config/step only: a warm-start snapshot, without optimizer or RNG continuation."""
    config = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg
    _atomic_write(path, lambda temporary: torch.save(
        {"model": model.state_dict(), "cfg": config, "step": step}, temporary))


def save_research_checkpoint(path, *, model, cfg, step, optimizer, scheduler, lr, history, provenance):
    """Trusted Python research state, retaining the historical field schema and adding Python RNG."""
    rng = capture_rng_state()
    state = {"model": model.state_dict(), "cfg": cfg, "step": step,
             "optimizer": optimizer.state_dict() if optimizer is not None else None,
             "scheduler": scheduler.state_dict() if scheduler is not None else None,
             "rng_python": rng["python"], "rng_numpy": rng["numpy"], "rng_torch": rng["torch"],
             "rng_cuda": rng.get("cuda"), "lr": lr, "history": history, "provenance": provenance}
    _atomic_write(path, lambda temporary: torch.save(state, temporary))


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


def save_rotating(base_path, cfg, model, logvars, step, opt, keep=3, *, dataconfigs=None, train_args=None, label_schema=None):
    """Keep numbered checkpoints and atomically refresh the convenient base-path copy."""
    try:
        if isinstance(keep, bool) or index(keep) <= 0:
            raise ValueError
        keep = index(keep)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError("checkpoint retention must be a positive integer") from exc
    stem, extension = _stem(base_path), _ext(base_path)
    path = f"{stem}.step{step}{extension}"
    save_checkpoint(path, cfg, model, logvars, step, opt, dataconfigs=dataconfigs, train_args=train_args, label_schema=label_schema)
    _atomic_write(base_path, lambda temporary: shutil.copyfile(path, temporary))
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
