"""A screen: a candidate architecture trained with the 1.0.0 recipe from the record cache, then compared by the assay
with the baseline's milestone at the same step. A yes or no on a candidate in under half an hour, several at once.

``run_screen(name, steps=..., ...)``:

1. **Refuses** before anything is written (FileNotFoundError, FileExistsError or ValueError, ConfigurationError
   among them): a name that is not one path component, a step the baseline has no milestone at, a cache that cannot
   feed the steps (``records.CachedRecordSource.require``), an unknown architecture or model option (the candidate's
   model setup is prepared), a suite not built in the dev build, and a screen name already in use.
2. Claims the screen directory ``out_root/<name>`` (one ``mkdir``: a name in use is refused, so two screens never
   share a directory) and copies the requested suites of the dev build into its own assay run directory,
   ``<screen>/assay``: every diagnosis is written there, never into the shared dev build.
3. Diagnoses the baseline milestone there first (``alignair_assay.workflow.diagnose.evaluate``: the head metrics and
   the end-to-end baseline; truth-substitution ``variants`` only when asked for), so any assay refusal comes before
   training.
4. Imports the candidates (``architectures``), then trains ``recipe.training_config`` from the cache with
   ``data_workers`` into ``<screen>/train`` (the trainer's log in ``<screen>/train.log``; a progress line here). Only
   the final ``model.alignair`` is kept: the trainer's rotating step checkpoints are removed once it finishes.
5. Diagnoses the candidate against the milestone, reusing the milestone's records, and writes the headline
   (``headline.render_headline``, with the report path and the wall time) to ``<screen>/headline.txt``.

``<screen>/screen.json`` holds the request, the cache and baseline used, the status (updated at each stage; a
failure is recorded before it is raised), the model keys, the report path, the times and the comparison summary.
Screens running at once share only inputs they read: the record cache, the dev build and the baseline milestones.
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from alignair_lab.headline import render_headline
from alignair_lab.recipe import (
    BASELINE_ARCHITECTURE,
    BATCH_SIZE,
    DATACONFIGS,
    POLICY,
    SEED,
    training_config,
)
from alignair_lab.records import CachedRecordSource

ARCH_ROOT = Path("/media/thomas/NVMe_Data/alignair_arch")
SCREENS = ARCH_ROOT / "screens"
CACHE = ARCH_ROOT / "cache" / "igh-1.0.0-seed0-40k"
DEV_BUILD = Path("/home/thomas/Desktop/AlignAIR/.private/arch/iter_dev")
MILESTONES = Path("/home/thomas/Desktop/AlignAIR/.private/arch/baseline_conv_v1/milestones")
DEFAULT_STEPS = 20_000
DEFAULT_VARIANTS = ("baseline",)
DEFAULT_DATA_WORKERS = 2
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MILESTONE = re.compile(r"model\.step(\d+)\.alignair")
PROGRESS_LINES = 20                       # progress lines printed over a training run


def milestone(milestones: Path, steps: int) -> Path:
    """The baseline's checkpoint at ``steps``; FileNotFoundError, naming the steps it has, when there is none."""
    path = Path(milestones) / f"model.step{steps}.alignair"
    if not path.is_file():
        held = sorted(int(match.group(1)) for match in map(_MILESTONE.fullmatch, os.listdir(milestones))
                      if match) if Path(milestones).is_dir() else []
        raise FileNotFoundError(f"the baseline has no milestone at step {steps}: {milestones} holds steps "
                                f"{', '.join(map(str, held)) or 'none'}")
    return path


def built_suites(dev_build: Path, suites: Sequence[str] | None) -> list[str]:
    """``suites``, each built in ``dev_build`` (ValueError otherwise); None: every suite built there, S1 to S12."""
    root = Path(dev_build) / "suites"
    if suites is None:
        names = [path.name for path in root.iterdir() if (path / "build.json").is_file()] if root.is_dir() else []
        if not names:
            raise ValueError(f"no suite is built in {dev_build}")
        return sorted(names, key=lambda name: (len(name), name))
    names = list(dict.fromkeys(suites))
    for name in names:
        if not NAME.fullmatch(name) or not (root / name / "build.json").is_file():
            raise ValueError(f"suite {name} is not built in {dev_build}")
    return names


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def run_screen(name: str, *, steps: int = DEFAULT_STEPS, architecture: str = BASELINE_ARCHITECTURE,
               model_options: Mapping | None = None, suites: Sequence[str] | None = None,
               max_cases: int | None = None, variants: Sequence[str] = DEFAULT_VARIANTS, device: str = "cuda",
               data_workers: int = DEFAULT_DATA_WORKERS, eval_batch_size: int = 256, cache=CACHE,
               dev_build=DEV_BUILD, milestones=MILESTONES, out_root=SCREENS,
               echo: Callable[[str], None] | None = None) -> dict:
    """Screen ``architecture`` with ``model_options`` for ``steps`` steps against the baseline milestone at that step
    (see the module docstring); returns the screen record (``screen.json``) with its printed ``headline``.

    ``echo`` receives the progress lines and the headline; by default they are printed to the standard output in
    effect at the call, which the trainer's redirected output does not reach."""
    start = time.perf_counter()
    if echo is None:
        stream = sys.stdout

        def echo(line: str) -> None:
            print(line, file=stream, flush=True)
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"a screen name is 1 to 64 letters, digits, '.', '_' or '-', starting with a letter or digit; "
                         f"got {name!r}")
    model_options = dict(model_options or {})
    baseline = milestone(Path(milestones), steps)
    source = CachedRecordSource(cache)
    manifest = source.require(DATACONFIGS, policy=POLICY, seed=SEED, records=steps * BATCH_SIZE)
    names = built_suites(Path(dev_build), suites)
    config = training_config(steps=steps, architecture=architecture, model_options=model_options, device=device,
                             data_workers=data_workers)
    _prepare_candidate(config)
    screen = Path(out_root) / name
    Path(out_root).mkdir(parents=True, exist_ok=True)
    try:
        screen.mkdir()
    except FileExistsError:
        raise FileExistsError(f"a screen named {name!r} exists ({screen}); choose another --name") from None

    record = {"name": name, "status": "preparing", "started": _now(), "steps": steps, "architecture": architecture,
              "model_options": model_options, "suites": names, "variants": list(variants), "max_cases": max_cases,
              "device": device, "data_workers": data_workers, "eval_batch_size": eval_batch_size,
              "recipe": {"dataconfigs": list(DATACONFIGS), "batch_size": BATCH_SIZE, "lr": config.lr, "seed": SEED},
              "cache": {"path": source.path, "steps": manifest["steps"], "records": manifest["records"],
                        "genairr_source_sha256": manifest["genairr"]["genairr_source_sha256"],
                        "recipe": manifest["recipe"], "code_sha256": manifest["code"]["sha256"],
                        "created": manifest["created"]},
              "baseline": {"architecture": BASELINE_ARCHITECTURE, "milestone": str(baseline)},
              "dev_build": str(Path(dev_build).resolve()), "screen_dir": str(screen.resolve())}

    def stage(status: str, **fields):
        record.update(status=status, **fields)
        _write_json(screen / "screen.json", record)
        echo(f"[screen {name}] {status}")

    try:
        layout = _assay_layout(screen / "assay", Path(dev_build), names)
        options = {"device": device, "batch_size": eval_batch_size, "variants": tuple(variants), "profile": False,
                   "max_cases": max_cases}
        stage("diagnosing the baseline milestone")
        from alignair_assay.workflow.diagnose import evaluate
        clock = time.perf_counter()
        base = evaluate(str(baseline), layout, names, **options)
        times = {"baseline_diagnosis": time.perf_counter() - clock}
        record["baseline"]["model_key"] = base.model_key
        _release_device(device)

        stage(f"training {architecture} for {steps} steps from the record cache")
        clock = time.perf_counter()
        trained = _train(config, screen, source, echo=lambda line: echo(f"[screen {name}] {line}"))
        times["training"] = time.perf_counter() - clock
        record["candidate"] = {"model_path": trained.model_path, "steps": trained.steps,
                               "steps_per_second": round(trained.steps / times["training"], 2)}
        _release_device(device)

        stage("diagnosing the candidate against the baseline")
        clock = time.perf_counter()
        result = evaluate(trained.model_path, layout, names, **options, against=str(baseline), label=name)
        times["candidate_diagnosis"] = time.perf_counter() - clock
        times["total"] = time.perf_counter() - start
        report = layout.diagnostics_md(result.model_key)
        headline = render_headline(result.against["summary"], title=_title(record, result, base.model_key))
        headline += "\n\n" + "\n".join(_footer(report, layout, result.model_key, base.model_key, times, record))
        (screen / "headline.txt").write_text(headline + "\n", encoding="utf-8")
        record["candidate"]["model_key"] = result.model_key
        record.update(report=str(report), seconds={key: round(value, 1) for key, value in times.items()},
                      comparison=result.against["summary"], finished=_now())
        stage("done")
        echo("")
        echo(headline)
        return {**record, "headline": headline}
    except BaseException as error:
        record.update(error=f"{type(error).__name__}: {error}", finished=_now())
        stage("failed")
        raise


def _prepare_candidate(config) -> None:
    """Import the candidates and prepare the candidate's model setup (no weights): an unknown architecture or model
    option is refused here, before anything is written."""
    import alignair_lab.architectures  # noqa: F401  (registers every candidate in this process)
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.core.components import architecture_components
    from alignair.reference.reference_set import ReferenceSet
    architecture_components(config.model_options.get("architecture", BASELINE_ARCHITECTURE))
    sources = [getattr(gd, source) for source in config.dataconfigs]
    prepare_model(ReferenceSet.from_dataconfigs(*sources), input_space=config.input_space,
                  label_space=config.label_space, **config.model_options)


def _assay_layout(root: Path, dev_build: Path, suites: Sequence[str]):
    """The screen's own assay run directory, holding copies of the dev build's ``suites``."""
    from alignair_assay.core.layout import RunLayout
    layout = RunLayout(root)
    for suite in suites:
        shutil.copytree(dev_build / "suites" / suite, layout.suite_dir(suite))
    return layout


def _train(config, screen: Path, source: CachedRecordSource, *, echo: Callable[[str], None]):
    """Train ``config`` from the cache into ``<screen>/train``, its log in ``<screen>/train.log``, echoing progress."""
    import alignair_lab.architectures  # noqa: F401  (the process that builds the model registers the candidates)
    from alignair import run_training
    every = max(1, config.steps // PROGRESS_LINES)
    clock = time.perf_counter()

    def progress(iteration):
        if iteration.step % every == 0 or iteration.step == config.steps:
            elapsed = time.perf_counter() - clock
            rate = iteration.step / elapsed
            left = (config.steps - iteration.step) / rate if rate else 0.
            echo(f"step {iteration.step}/{config.steps}, {rate:.1f} steps/s, {_minutes(left)} left")

    with open(screen / "train.log", "w", buffering=1, encoding="utf-8") as log, contextlib.redirect_stdout(log):
        trained = run_training(config, output_dir=screen / "train", record_source=source, on_iteration=progress)
    for rotated in (screen / "train").glob("model.step*.alignair"):    # model.alignair holds the last step
        rotated.unlink()
    return trained


def _release_device(device: str) -> None:
    """Return this process's cached accelerator memory, which other runs on the GPU may need."""
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()


def _title(record: dict, result, baseline_key: str) -> list[str]:
    options = f" with {json.dumps(record['model_options'])}" if record["model_options"] else ""
    return [f"Screen {record['name']}: {record['architecture']}{options}, {record['steps']} steps, against the "
            f"baseline {BASELINE_ARCHITECTURE} at step {record['steps']}.",
            f"Candidate {result.model_key} ({record['candidate']['model_path']}); baseline {baseline_key} "
            f"({record['baseline']['milestone']}).", ""]


def _footer(report: Path, layout, key: str, baseline_key: str, times: dict, record: dict) -> list[str]:
    detail = layout.diagnostics_dir(key) / "<S>" / f"against_{baseline_key}.md"
    return [f"Report: {report}", f"Per-suite detail: {detail}",
            f"Wall time: {_minutes(times['total'])} (baseline diagnosis {_minutes(times['baseline_diagnosis'])}, "
            f"training {_minutes(times['training'])} at {record['candidate']['steps_per_second']} steps/s, "
            f"candidate diagnosis {_minutes(times['candidate_diagnosis'])})."]
