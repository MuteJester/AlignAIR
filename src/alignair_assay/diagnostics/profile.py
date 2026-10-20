"""Where AlignAIR spends its time: per network submodule and per post-network stage.

``profile_model`` replays AlignAIR's predict pipeline (``replay.replay``) on the given reads and reports two
breakdowns, each as milliseconds per read and share of its total:

- ``modules``: the network's leaf submodules, timed with forward hooks: ``embedding`` (called twice per forward, the
  second time on the orientation-corrected read; both calls are summed), ``orientation_head``, ``meta_tower``,
  ``meta_heads.<name>``, and per gene ``<g>.seg_tower``, ``<g>.start_head``, ``<g>.end_head``, ``<g>.cutout``,
  ``<g>.cls_tower``, ``<g>.cls_mid`` and ``<g>.cls_head``. ``other`` is the rest of the forward stage (the
  orientation transform, soft-argmax, tokenizing and the copies to the host): the forward stage's time not spent
  inside a hooked module. Its shares sum to 1 over the forward stage's time.
- ``stages``: the replay's stages (``replay.STAGES``), from ``StageOutputs.seconds``; ``forward`` is the whole
  network call. Its shares sum to 1 over the total.

On CUDA the hooks synchronize the device before and after each hooked module so the times are the kernels', and a
warm-up pass runs first. The synchronizations cost time, so the module timings come from separate, hooked runs; the
stage timings and the reads per second come from unhooked runs of the same reads. Each entry is the median over
``repeats`` runs.

Profiling never shares a call with an accuracy replay: the hooks are installed for the hooked runs only and removed
in a ``finally``, and nothing the profile computes feeds back into the model (the tests pin that the rows of a
replay are identical before and after a profile).
"""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager

import torch

from alignair_assay.diagnostics.replay import STAGES, replay

# The forward time not spent inside a hooked module.
OTHER = "other"


def profile_model(model, reference_set, sequences: list[str], *, device: str, batch_size: int,
                  warmup_batches: int = 1, repeats: int = 3) -> dict:
    """The speed profile of ``model`` on ``sequences`` (see the module docstring), as a dict:

    ``device``, ``batch_size``, ``reads``, ``repeats``, ``reads_per_second`` (the unhooked replay's overall rate),
    ``seconds_per_run`` (its median wall time), ``modules`` and ``stages`` (name -> ``{"ms_per_read", "share"}``).
    The warm-up replays the first ``warmup_batches * batch_size`` reads. ValueError for no sequence, a
    ``batch_size`` or ``repeats`` below 1, or a negative ``warmup_batches``."""
    if not sequences:
        raise ValueError("no sequences to profile")
    if batch_size < 1 or repeats < 1 or warmup_batches < 0:
        raise ValueError(f"batch_size ({batch_size}) and repeats ({repeats}) must be at least 1 and warmup_batches "
                         f"({warmup_batches}) at least 0")
    model = model.to(device)
    ids = [f"profile{i}" for i in range(len(sequences))]
    n = len(sequences)

    def run():
        return replay(model, reference_set, sequences, ids, device=device, batch_size=batch_size)

    if warmup_batches:
        warm = sequences[:warmup_batches * batch_size]
        replay(model, reference_set, warm, ids[:len(warm)], device=device, batch_size=batch_size)
        _sync(device)

    stage_runs = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        seconds = run().seconds
        _sync(device)
        stage_runs.append((time.perf_counter() - start, seconds))
    wall = statistics.median(total for total, _ in stage_runs)
    stage_seconds = {stage: statistics.median(seconds[stage] for _, seconds in stage_runs) for stage in STAGES}

    module_runs = []
    with _hooked(model, device) as spent:
        for _ in range(repeats):
            spent.clear()
            forward_seconds = run().seconds["forward"]
            module_runs.append((dict(spent), forward_seconds))
    module_seconds = {name: statistics.median(times.get(name, 0.0) for times, _ in module_runs)
                      for name in module_runs[0][0]}
    module_seconds[OTHER] = max(statistics.median(forward - sum(times.values()) for times, forward in module_runs),
                                0.0)

    return {"device": device, "batch_size": batch_size, "reads": n, "repeats": repeats,
            "reads_per_second": n / wall, "seconds_per_run": wall,
            "modules": _breakdown(module_seconds, n), "stages": _breakdown(stage_seconds, n)}


def _breakdown(seconds: dict[str, float], reads: int) -> dict[str, dict[str, float]]:
    total = sum(seconds.values())
    return {name: {"ms_per_read": 1000.0 * value / reads, "share": value / total if total else 0.0}
            for name, value in seconds.items()}


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def hooked_modules(model) -> dict[str, torch.nn.Module]:
    """The leaf submodules that are timed, by profile name (the module docstring lists them)."""
    modules = {"embedding": model.embedding, "orientation_head": model.orientation_head,
               "meta_tower": model.meta_tower}
    for key, head in model.meta_heads.items():
        modules[f"meta_heads.{key}"] = head
    if getattr(model, "state_branch", None) is not None:
        modules["state_branch"] = model.state_branch
    for gene, branch in model.branches.items():
        for part in ("seg_tower", "start_head", "end_head", "cutout", "cls_tower", "cls_mid", "cls_head"):
            modules[f"{gene}.{part}"] = getattr(branch, part)
    return modules


@contextmanager
def _hooked(model, device: str):
    """Forward hooks that add each hooked module's time (summed over its calls) to the yielded dict; removed on
    exit, whatever happens."""
    spent: dict[str, float] = {}
    started: dict[str, list[float]] = {}
    handles = []

    def make(name: str):
        def before(module, args):
            _sync(device)
            started.setdefault(name, []).append(time.perf_counter())

        def after(module, args, output):
            _sync(device)
            spent[name] = spent.get(name, 0.0) + time.perf_counter() - started[name].pop()

        return before, after

    try:
        for name, module in hooked_modules(model).items():
            before, after = make(name)
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
        yield spent
    finally:
        for handle in handles:
            handle.remove()
