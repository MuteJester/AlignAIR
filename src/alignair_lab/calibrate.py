"""Calibration: the run-to-run training noise of a screen protocol, measured with baseline screens of several init
seeds.

A candidate's Δ against the registered baseline screen holds training noise beside the candidate's effect. Baseline
screens of other init seeds, trained on the same records, differ by that noise alone, so their pairwise Δs bound it:
the null band (``noise.null_band``) a candidate's headline weighs its Δs against (``baselines.null_band``).

``run_calibration(steps=..., init_seeds=(0, 1, 2), ...)``:

1. **Refuses** before anything runs (ValueError, FileExistsError, ConfigurationError among them): fewer than two init
   seeds, one given twice or one that is not a uint32; a parallel count below 1; an unknown schedule; a calibration
   name that is not one path component, or in use; for an init seed, several finished baseline screens of the
   protocol, or none and a screen in the way of the one to train (``baselines.screen_name``); an unknown model option; a
   cache that cannot feed the steps (``records.CachedRecordSource.require``); a suite not built in the dev build.
2. Claims the calibration directory ``out_root/<name>`` (default ``calibration_name``) and records the request and
   its progress in its ``calibration.json`` (a failure is recorded before it is raised).
3. For each init seed, reuses the finished baseline screen of the protocol, the model options and that seed
   (``baselines.baseline_screens``), or trains one with ``python -m alignair_lab screen`` under the GenAIRR and the code
   this process imports, at most ``parallel`` at once, each one's output in ``<calibration>/<screen>.log``. A failed
   run fails the calibration once the runs under way have finished: those finished are reused by the next.
4. Checks every baseline screen's checkpoint (``baselines.check_baseline``, ``baselines.same_genairr``), then compares
   each pair of them, the later init seed's against the earlier's, with the assay in its own run directory,
   ``<calibration>/assay``, in this process: under one AlignAIR code.
5. Writes ``<calibration>/null_band.json``: the protocol, the model options, the suites, their builds (the SHA-256 of
   each ``build.json``) and the cases compared, the
   screens, each pair's pooled Δs and each suite's (``noise.pair``) and the null band; each baseline seed's values
   (``seeds``: pooled and per suite, from its pair with the registered seed, ``relative_to``; ``noise.seed_values``)
   and their mean and SD per metric (``baseline``), the distribution a candidate's seeds are compared with
   (``noise.decide``); and prints the band of each headline metric. A null band written before the per-seed values
   gives them from its pairs (``baselines.calibration_seeds``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path

from alignair.errors import ConfigurationError
from alignair_lab.headline import HEADLINE
from alignair_lab.noise import baseline_summary, null_band, pair, seed_values
from alignair_lab.secondary import METRIC, cached_comparison, merged
from alignair_lab.recipe import BATCH_SIZE, DATACONFIGS, POLICY, SEED, schedule as learning_schedule, training_config
from alignair_lab.records import CachedRecordSource
from alignair_lab.baselines import (
    BASELINE_INIT_SEED,
    CHECKPOINT,
    NAME,
    NULL_ARCHITECTURE,
    NULL_BAND,
    NULL_BAND_KIND,
    NULL_BAND_VERSION,
    baseline_screens,
    check_baseline,
    check_init_seed,
    now,
    protocol,
    same_genairr,
    screen_name,
    write_json,
)
from alignair_lab.screen import (
    CACHE,
    DEFAULT_DATA_WORKERS,
    DEFAULT_SCHEDULE,
    DEFAULT_STEPS,
    DEV_BUILD,
    SCREENS,
    assay_layout,
    built_suites,
    prepare_candidate,
    suite_builds,
)

DEFAULT_INIT_SEEDS = (0, 1, 2)
VARIANTS = ("baseline",)                  # the end-to-end metrics, the headline's clone and junction among them
RECORD = "calibration.json"
POLL_SECONDS = 1.


def calibration_name(cache, steps: int, schedule: str) -> str:
    """The calibration directory's default name."""
    return f"calibration-{Path(cache).name}-s{steps}-{schedule}"


def _seeds(init_seeds: Sequence[int]) -> list[int]:
    seeds = [check_init_seed(seed) for seed in init_seeds]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError(f"calibrate compares two init seeds or more, each once; got {seeds}")
    return seeds


def _command(name: str, seed: int, *, steps: int, schedule: str, model_options: Mapping, device: str,
             data_workers: int, cache: str, out_root: Path) -> list[str]:
    options = [part for key, value in model_options.items()
               for part in ("--model-option", f"{key}={json.dumps(value)}")]
    return [sys.executable, "-m", "alignair_lab", "screen", "--name", name, "--steps", str(steps), "--architecture",
            NULL_ARCHITECTURE, "--init-seed", str(seed), "--schedule", schedule, *options, "--device", device,
            "--data-workers", str(data_workers), "--cache", cache, "--out-root", str(out_root)]


def _environment() -> dict:
    """This process's environment, with the GenAIRR and the AlignAIR code it imports first on the path."""
    import GenAIRR
    import alignair_lab
    first = [str(Path(GenAIRR.__file__).resolve().parents[1]), str(Path(alignair_lab.__file__).resolve().parents[1])]
    rest = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys([*first, *rest]))}


def train_screens(plan: list[tuple[int, str, list[str]]], directory: Path, parallel: int,
                  echo: Callable[[str], None], *, label: str = "calibrate", what: str = "baseline screen") -> None:
    """Run each ``(seed, name, command)`` of ``plan``, at most ``parallel`` at once, its output in
    ``directory/<name>.log``, echoing ``[label]`` progress lines that call each run a ``what``; RuntimeError, naming
    the logs, once the runs under way have finished if any failed. Interrupted, it ends the runs under way."""
    pending, running, failed = list(plan), {}, []
    environment = _environment()
    try:
        while pending or running:
            while pending and len(running) < parallel and not failed:
                seed, name, command = pending.pop(0)
                log = open(directory / f"{name}.log", "w", encoding="utf-8")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                running[process] = (seed, name, log, time.perf_counter())
                echo(f"[{label}] training the {what} {name} (init seed {seed}); output: {log.name}")
            for process in [process for process in running if process.poll() is not None]:
                seed, name, log, begun = running.pop(process)
                log.close()
                if process.returncode:
                    failed.append(log.name)
                    echo(f"[{label}] the {what} {name} failed (exit {process.returncode}): see {log.name}")
                else:
                    echo(f"[{label}] trained {name} in {(time.perf_counter() - begun) / 60:.1f} min")
            if running:
                time.sleep(POLL_SECONDS)
            elif failed:
                break
    finally:
        for process, (_, _, log, _) in running.items():     # interrupted: no run outlives the calibration
            process.terminate()
            process.wait()
            log.close()
    if failed:
        raise RuntimeError(f"{len(failed)} {what}(s) failed, see {', '.join(failed)}; the finished ones are reused "
                           f"when {label} runs again")


def run_calibration(*, steps: int = DEFAULT_STEPS, init_seeds: Sequence[int] = DEFAULT_INIT_SEEDS, parallel: int = 1,
                    schedule: str = DEFAULT_SCHEDULE, model_options: Mapping | None = None,
                    suites: Sequence[str] | None = None, max_cases: int | None = None, device: str = "cuda",
                    data_workers: int = DEFAULT_DATA_WORKERS, eval_batch_size: int = 256, cache=CACHE,
                    dev_build=DEV_BUILD, out_root=SCREENS, name: str | None = None,
                    echo: Callable[[str], None] | None = None) -> dict:
    """Calibrate the protocol of ``steps`` steps under ``schedule`` from ``cache`` with baseline screens of
    ``init_seeds`` and ``model_options`` (see the module docstring); returns the null band record
    (``null_band.json``). ``echo`` receives the progress lines and the summary (default: printed)."""
    if echo is None:
        def echo(line: str) -> None:
            print(line, flush=True)
    seeds = _seeds(init_seeds)
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
        raise ValueError(f"parallel is a positive integer, got {parallel!r}")
    learning = learning_schedule(schedule, steps)
    model_options = json.loads(json.dumps(dict(model_options or {})))
    source = CachedRecordSource(cache)
    wanted = protocol(source.manifest, steps, learning)
    out_root = Path(out_root)
    name = calibration_name(source.path, steps, schedule) if name is None else name
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"a calibration name is 1 to 64 letters, digits, '.', '_' or '-', starting with a letter or "
                         f"digit; got {name!r}")
    directory = out_root / name
    if directory.exists():
        raise FileExistsError(f"a calibration named {name!r} exists ({directory}); choose another --name")
    screens, plan = {}, []
    for seed in seeds:
        done = [(found, record) for found, record in baseline_screens(out_root, wanted, model_options, seed)
                if record.get("status") == "done"]
        if len(done) > 1:
            raise ConfigurationError(f"{len(done)} baseline screens of init seed {seed} and this protocol in "
                                     f"{out_root}: {', '.join(found.name for found, _ in done)}",
                                     context={"field": "init_seeds", "path": str(out_root)},
                                     hint="Keep one baseline screen of each init seed: move the others away.")
        if done:
            screens[str(seed)] = {"name": done[0][0].name, "reused": True}
            continue
        planned = screen_name(source.path, steps, schedule, seed)
        if (out_root / planned).exists():
            raise FileExistsError(f"the baseline screen of init seed {seed} would be {out_root / planned}, which "
                                  "exists and is not a finished baseline screen of this protocol: remove it")
        screens[str(seed)] = {"name": planned, "reused": False}
        plan.append((seed, planned, _command(planned, seed, steps=steps, schedule=schedule,
                                             model_options=model_options, device=device, data_workers=data_workers,
                                             cache=source.path, out_root=out_root)))
    prepare_candidate(training_config(steps=steps, architecture=NULL_ARCHITECTURE, model_options=model_options,
                                      schedule=learning))
    manifest = source.require(DATACONFIGS, policy=POLICY, seed=SEED, records=steps * BATCH_SIZE)
    names = built_suites(Path(dev_build), suites)
    builds = suite_builds(Path(dev_build), names)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError:
        raise FileExistsError(f"a calibration named {name!r} exists ({directory}); choose another --name") from None

    record = {"kind": NULL_BAND_KIND, "version": NULL_BAND_VERSION, "name": name, "status": "preparing",
              "started": now(), "protocol": wanted, "options": model_options, "suites": names,
              "suite_builds": builds, "max_cases": max_cases,
              "variants": list(VARIANTS), "device": device, "eval_batch_size": eval_batch_size, "init_seeds": seeds,
              "cache": {"path": source.path, "identity": wanted["cache"]}, "screens": screens}

    def stage(status: str, **fields):
        record.update(status=status, **fields)
        write_json(directory / RECORD, record)
        echo(f"[calibrate] {status}")

    try:
        stage(f"training {len(plan)} baseline screen(s), reusing {len(seeds) - len(plan)}")
        train_screens(plan, directory, parallel, echo)
        for seed in seeds:
            found = out_root / screens[str(seed)]["name"]
            provenance = check_baseline(found / CHECKPOINT, steps, schedule=learning, options=model_options,
                                        architecture=NULL_ARCHITECTURE, init_seed=seed)
            same_genairr(provenance, manifest)
            screens[str(seed)]["checkpoint"] = provenance["checkpoint"]
        stage(f"comparing the {len(seeds) * (len(seeds) - 1) // 2} pairs of baseline screens")
        from alignair_assay.workflow.diagnose import evaluate
        layout = assay_layout(directory / "assay", Path(dev_build), names)
        pairs, code = [], None
        for first, second in combinations(seeds, 2):
            earlier, later = screens[str(first)], screens[str(second)]
            result = evaluate(later["checkpoint"], layout, names, device=device, batch_size=eval_batch_size,
                              variants=VARIANTS, profile=False, max_cases=max_cases, against=earlier["checkpoint"],
                              label=later["name"])
            summary = result.against["summary"]
            earlier["model_key"], later["model_key"] = summary["baseline"], result.model_key
            code = summary["code"]["this"]
            extra = cached_comparison(layout.diagnostics_root, result.model_key, summary["baseline"], METRIC)
            pairs.append({"a": first, "b": second, **pair(merged(summary, extra))})
        band = null_band(pairs)
        relative_to = BASELINE_INIT_SEED if BASELINE_INIT_SEED in seeds else seeds[0]
        values = seed_values(pairs, seeds, relative_to=relative_to)
        record.update(status="done", pairs=pairs, band=band, relative_to=relative_to, seeds=values,
                      baseline=baseline_summary(values), code_sha256=code, finished=now())
        write_json(directory / NULL_BAND, record)
        stage("done")
        shown = [(label, band[metric]) for metric, label in HEADLINE if metric in band]
        echo(f"[calibrate] the null band, the largest |pooled Δ| of {len(pairs)} pairs, in pp: "
             + ", ".join(f"{label} {100 * value:.2f}" for label, value in shown)
             + f"; every metric's and each pair's Δs are in {directory / NULL_BAND}")
        return record
    except BaseException as error:
        record.update(error=f"{type(error).__name__}: {error}", finished=now())
        stage("failed")
        raise
