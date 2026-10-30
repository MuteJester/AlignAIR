"""A screen: an architecture trained with the recipe (``recipe``) from the record cache, then compared by the assay with
a baseline at the same step. A yes or no on a candidate in under half an hour, several at once.

**Baseline screens.** A screen of ``baselines.NULL_ARCHITECTURE`` (lab-null-v1: conv-v1 under a lab name) compared
with nothing is a baseline screen: one replicate of the baseline, of its ``init_seed``. A screen of another
architecture compares with the registered baseline screen of its protocol (``baselines``: the recipe id, the record
cache's content identity, the steps and the schedule), or with the baseline screen it names (``against_screen``).
Several baseline screens of other init seeds measure the run-to-run noise (``calibrate``), whose null band the
candidate's headline weighs its Δs against. A screen compares with a milestone of the baseline's own run instead when
told one (``against_milestone``, Tier 2): the milestone's run kept a constant rate.

By default a screen cools its learning rate down (``recipe.schedule``: cosine to a hundredth of it over its steps), so
that replicate runs end close to each other, and draws its initial weights from init seed 0
(``TrainingConfig.init_seed``): every screen's batches are then the same, whatever its model's build draws.

``run_screen(name, steps=..., ...)``:

1. **Refuses** before anything is written (FileNotFoundError, FileExistsError or ValueError, ConfigurationError
   among them): a name that is not one path component; an unknown schedule or an init seed that is not a uint32; a
   baseline that is not the recipe's at that step under that schedule, or that continued another checkpoint, clipped
   its gradients or ran on another GenAIRR than the cache's (``baselines.check_baseline``, ``same_genairr``: a
   baseline is trusted by its checkpoint's contents, not its file name or its screen's record): a milestone, a
   baseline screen named that is not a finished baseline screen of the protocol, and no registered baseline screen
   (the error says the commands that make one), or several; two null bands of the protocol; a cache that cannot feed
   the steps or does not reproduce the live stream (``records.CachedRecordSource.require``); an unknown architecture
   or model option (the candidate's model setup is prepared); a suite not built in the dev build; and a screen name
   already in use.
2. Claims the screen directory ``out_root/<name>`` (one ``mkdir``: a name in use is refused, so two screens never
   share a directory). A screen that compares copies the requested suites of the dev build into its own assay run
   directory, ``<screen>/assay``: every diagnosis is written there, never into the shared dev build.
3. Diagnoses the baseline's checkpoint there first (``alignair_assay.workflow.diagnose.evaluate``: the head metrics
   and the end-to-end baseline; truth-substitution ``variants`` only when asked for), so any assay refusal comes
   before training. The candidate is then diagnosed in the same process: both under one AlignAIR code.
4. Imports the candidates (``architectures``), then trains ``recipe.training_config`` from the cache with
   ``data_workers`` into ``<screen>/train`` (the trainer's log in ``<screen>/train.log``; a progress line here). Only
   the final ``model.alignair`` is kept: the trainer's rotating step checkpoints are removed once it finishes.
5. Diagnoses the candidate against the baseline, reusing the baseline's records, and writes the headline
   (``headline.render_headline`` with the null band, the report path and the wall time) to ``<screen>/headline.txt``.
   A baseline screen diagnoses nothing: its headline names what it is.

``<screen>/screen.json`` holds the request (its ``role``: ``baseline`` or ``candidate``, its ``init_seed``,
``schedule`` and ``protocol``), the cache and the baseline used (its kind, ``screen`` or ``milestone``, step, recipe,
git commit and GenAIRR identity, also on the headline), the null band used, the status (updated at each stage; a
failure is recorded before it is raised), the model keys, the report path, the times, the comparison summary and the
noise ``decisions``. Screens running at once share only inputs they read: the record cache, the dev build, the
baselines and the null bands.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from alignair.train.trainer import RECIPE
from alignair_lab.baselines import (
    BASELINE_INIT_SEED,
    NULL_ARCHITECTURE,
    SCREEN_RECORD,
    NAME,
    baseline_milestone,
    baseline_screen,
    check_init_seed,
    now,
    null_band,
    option_arguments,
    protocol,
    same_genairr,
    write_json,
)
from alignair_lab.headline import HEADLINE, render_headline
from alignair_lab.noise import decide
from alignair_lab.recipe import (
    BASELINE_ARCHITECTURE,
    BATCH_SIZE,
    DATACONFIGS,
    POLICY,
    SEED,
    schedule as learning_schedule,
    training_config,
)
from alignair_lab.records import CachedRecordSource

ARCH_ROOT = Path("/media/thomas/NVMe_Data/alignair_arch")
SCREENS = ARCH_ROOT / "screens"
CACHE = ARCH_ROOT / "cache" / "igh-v3-seed0-40k"
DEV_BUILD = Path("/home/thomas/Desktop/AlignAIR/.private/arch/iter_dev")
MILESTONES = Path("/home/thomas/Desktop/AlignAIR/.private/arch/baseline_conv_v1/milestones")
DEFAULT_STEPS = 20_000
DEFAULT_VARIANTS = ("baseline",)
DEFAULT_DATA_WORKERS = 2
DEFAULT_SCHEDULE = "cosine"
DEFAULT_INIT_SEED = 0
PROGRESS_LINES = 20                       # progress lines printed over a training run




def suite_builds(dev_build: Path, suites: Sequence[str]) -> dict[str, str]:
    """Each of ``suites`` -> the SHA-256 of its ``build.json`` in ``dev_build``: a null band is of one build."""
    return {suite: hashlib.sha256((Path(dev_build) / "suites" / suite / "build.json").read_bytes()).hexdigest()
            for suite in suites}


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




def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def run_screen(name: str, *, steps: int = DEFAULT_STEPS, architecture: str = BASELINE_ARCHITECTURE,
               model_options: Mapping | None = None, suites: Sequence[str] | None = None,
               max_cases: int | None = None, variants: Sequence[str] = DEFAULT_VARIANTS, device: str = "cuda",
               data_workers: int = DEFAULT_DATA_WORKERS, eval_batch_size: int = 256, cache=CACHE,
               dev_build=DEV_BUILD, out_root=SCREENS, schedule: str = DEFAULT_SCHEDULE,
               init_seed: int = DEFAULT_INIT_SEED, against_screen: str | None = None, against_milestone=None,
               baseline_options: Mapping | None = None, echo: Callable[[str], None] | None = None) -> dict:
    """Screen ``architecture`` with ``model_options`` for ``steps`` steps under ``schedule`` from ``init_seed``'s
    initial weights (see the module docstring); returns the screen record (``screen.json``) with its printed
    ``headline``.

    A screen of ``NULL_ARCHITECTURE`` given no baseline is a baseline screen. Any other compares with the baseline
    screen ``against_screen``, or with the registered baseline screen of its protocol, or, given the directory
    ``against_milestone``, with the milestone there at ``steps``. The baseline is of its architecture's default
    options, or of ``baseline_options`` (recorded in ``screen.json`` and on the headline). ``echo`` receives the
    progress lines and the headline; by default they are printed to the standard output in effect at the call,
    which the trainer's redirected output does not reach."""
    start = time.perf_counter()
    if echo is None:
        stream = sys.stdout

        def echo(line: str) -> None:
            print(line, file=stream, flush=True)
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"a screen name is 1 to 64 letters, digits, '.', '_' or '-', starting with a letter or digit; "
                         f"got {name!r}")
    if against_screen is not None and against_milestone is not None:
        raise ValueError("compare with a baseline screen or with a milestone, not both")
    init_seed = None if init_seed is None else check_init_seed(init_seed)
    learning = learning_schedule(schedule, steps)
    model_options, baseline_options = dict(model_options or {}), dict(baseline_options or {})
    role = ("baseline" if architecture == NULL_ARCHITECTURE and against_screen is None and against_milestone is None
            else "candidate")
    if role == "baseline" and baseline_options:
        raise ValueError("a baseline screen compares with nothing: it takes no baseline options")
    source = CachedRecordSource(cache)
    wanted = protocol(source.manifest, steps, learning)
    if role == "baseline":
        provenance = None
    elif against_milestone is not None:
        provenance = baseline_milestone(Path(against_milestone), steps, baseline_options, schedule=learning)
    else:
        provenance = baseline_screen(Path(out_root), wanted, baseline_options, name=against_screen, cache=source.path,
                                     schedule=schedule)
    manifest = source.require(DATACONFIGS, policy=POLICY, seed=SEED, records=steps * BATCH_SIZE)
    names = builds = band = None
    if provenance is not None:
        same_genairr(provenance, manifest)
        names = built_suites(Path(dev_build), suites)
        builds = suite_builds(Path(dev_build), names)
    if provenance is not None and provenance["kind"] == "screen":       # a milestone's run is another regime of noise
        band = null_band(Path(out_root), wanted, baseline_options, names, max_cases, builds)
    config = training_config(steps=steps, architecture=architecture, model_options=model_options, device=device,
                             data_workers=data_workers, schedule=learning, init_seed=init_seed)
    prepare_candidate(config)
    screen = Path(out_root) / name
    Path(out_root).mkdir(parents=True, exist_ok=True)
    try:
        screen.mkdir()
    except FileExistsError:
        raise FileExistsError(f"a screen named {name!r} exists ({screen}); choose another --name") from None

    record = {"name": name, "status": "preparing", "started": now(), "role": role, "steps": steps,
              "architecture": architecture, "model_options": model_options, "init_seed": init_seed,
              "schedule": schedule, "protocol": wanted, "suites": names, "suite_builds": builds,
              "variants": list(variants),
              "max_cases": max_cases, "device": device, "data_workers": data_workers,
              "eval_batch_size": eval_batch_size,
              "recipe": {"id": RECIPE, "dataconfigs": list(DATACONFIGS), "batch_size": BATCH_SIZE, "lr": config.lr,
                         "seed": SEED},
              "cache": {"path": source.path, "identity": wanted["cache"], "steps": manifest["steps"],
                        "records": manifest["records"],
                        "genairr_source_sha256": manifest["genairr"]["genairr_source_sha256"],
                        "genairr_version": manifest["genairr"].get("genairr_version"), "recipe": manifest["recipe"],
                        "code_sha256": manifest["code"]["sha256"], "created": manifest["created"],
                        "verification": source.verification},
              "baseline": provenance, "null_band": band,
              "dev_build": str(Path(dev_build).resolve()), "out_root": str(Path(out_root).resolve()),
              "screen_dir": str(screen.resolve())}

    def stage(status: str, **fields):
        record.update(status=status, **fields)
        write_json(screen / SCREEN_RECORD, record)
        echo(f"[screen {name}] {status}")

    try:
        times = {}
        if provenance is not None:
            layout = assay_layout(screen / "assay", Path(dev_build), names)
            options = {"device": device, "batch_size": eval_batch_size, "variants": tuple(variants), "profile": False,
                       "max_cases": max_cases}
            stage(f"diagnosing the baseline {provenance['kind']}")
            from alignair_assay.workflow.diagnose import evaluate
            clock = time.perf_counter()
            base = evaluate(provenance["checkpoint"], layout, names, **options)
            times["baseline_diagnosis"] = time.perf_counter() - clock
            record["baseline"]["model_key"] = base.model_key
            _release_device(device)

        stage(f"training {architecture} for {steps} steps from the record cache")
        clock = time.perf_counter()
        trained = _train(config, screen, source, echo=lambda line: echo(f"[screen {name}] {line}"))
        times["training"] = time.perf_counter() - clock
        record["candidate"] = {"model_path": trained.model_path, "steps": trained.steps,
                               "steps_per_second": round(trained.steps / times["training"], 2)}
        _release_device(device)

        if provenance is None:
            times["total"] = time.perf_counter() - start
            headline = "\n".join(_baseline_headline(record, times))
        else:
            stage("diagnosing the candidate against the baseline")
            clock = time.perf_counter()
            result = evaluate(trained.model_path, layout, names, **options, against=provenance["checkpoint"],
                              label=name)
            times["candidate_diagnosis"] = time.perf_counter() - clock
            times["total"] = time.perf_counter() - start
            summary = result.against["summary"]
            report = layout.diagnostics_md(result.model_key)
            limits = None if band is None else band["band"]
            title = _title(record, result, base.model_key, summary["code"]["this"])
            headline = render_headline(summary, title=title, null_band=limits)
            headline += "\n\n" + "\n".join(_footer(report, layout, result.model_key, base.model_key, times, record))
            record["candidate"]["model_key"] = result.model_key
            record.update(report=str(report), comparison=summary,
                          decisions=decide(summary, limits, [metric for metric, _ in HEADLINE]))
        (screen / "headline.txt").write_text(headline + "\n", encoding="utf-8")
        record.update(seconds={key: round(value, 1) for key, value in times.items()}, finished=now())
        stage("done")
        echo("")
        echo(headline)
        return {**record, "headline": headline}
    except BaseException as error:
        record.update(error=f"{type(error).__name__}: {error}", finished=now())
        stage("failed")
        raise


def prepare_candidate(config) -> None:
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


def assay_layout(root: Path, dev_build: Path, suites: Sequence[str]):
    """An assay run directory at ``root``, holding copies of the dev build's ``suites``."""
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


def _seed(init_seed) -> str:
    return "none (its data seed)" if init_seed is None else str(init_seed)


def _training_text(record: dict) -> str:
    options = f" with {json.dumps(record['model_options'])}" if record["model_options"] else ""
    return (f"{record['architecture']}{options}, {record['steps']} steps ({record['schedule']} schedule, init seed "
            f"{_seed(record['init_seed'])})")


def _title(record: dict, result, baseline_key: str, code: str | None) -> list[str]:
    baseline = record["baseline"]
    what = "baseline screen" if baseline["kind"] == "screen" else "milestone"
    against = (f"the baseline screen {baseline['name']} ({NULL_ARCHITECTURE}, init seed {_seed(baseline['init_seed'])})"
               if baseline["kind"] == "screen" else f"the baseline {BASELINE_ARCHITECTURE}'s milestone")
    seeds = ([f"Note: the {what} drew its initial weights from init seed {_seed(baseline['init_seed'])}, this screen "
              f"from {_seed(record['init_seed'])}: their initial weights differ, and so do their batches' orientation "
              "draws."] if baseline["init_seed"] != record["init_seed"] else [])
    return [f"Screen {record['name']}: {_training_text(record)}, against {against} at step {record['steps']}.",
            f"Candidate {result.model_key} ({record['candidate']['model_path']}); baseline {baseline_key} "
            f"({baseline['checkpoint']}).", _provenance_line(baseline), *seeds, _cache_line(record["cache"]),
            *_band_lines(record, code), ""]


def _provenance_line(baseline: dict) -> str:
    commit = (baseline.get("git_commit") or "unknown")[:12]
    references = ", ".join((schema or "unknown")[:12] for schema in baseline["reference_schema_sha256"])
    source = (f", source {baseline['genairr_source_sha256'][:12]}" if baseline.get("genairr_source_sha256")
              else "")
    options = (f"; baseline options {json.dumps(baseline['options'])}" if baseline.get("options")
               else "; its architecture's default options")
    what = (f"Baseline screen {baseline['name']}" if baseline["kind"] == "screen" else "Baseline milestone")
    return (f"{what}: saved at step {baseline['step']} by recipe {baseline['recipe']}, at git {commit}, with GenAIRR "
            f"{baseline.get('genairr_version') or 'unknown'}{source} (reference {references}), on "
            f"{baseline.get('saved') or 'an unrecorded date'}{options}.")


def _cache_line(cache: dict) -> str:
    verification = cache.get("verification") or {}
    if verification.get("code_changed"):
        return (f"Record cache: producing code changed since the write; deep-verified {verification['records']} "
                "records live.")
    return (f"Record cache: producing code unchanged since the write; verified {verification.get('records', 0)} "
            "records live.")


def _band_lines(record: dict, code: str | None) -> list[str]:
    """The null band's line: its path, or why there is none (with the calibration that makes the band this screen
    would use); and a warning when it was measured under other AlignAIR code than this comparison."""
    band = record["null_band"]
    if record["baseline"]["kind"] == "milestone":
        return ["Null band: none applies, so uncalibrated (Tier 2: milestone): a null band measures the baseline "
                "screens of a protocol, not a long run's milestone."]
    if band is None:
        cases = "" if record["max_cases"] is None else f" --max-cases {record['max_cases']}"
        return [f"Null band: none of this protocol, options, suite build and cases, so uncalibrated (python -m "
                f"alignair_lab calibrate --steps {record['steps']} --init-seeds 0,1,2 --schedule {record['schedule']} "
                f"--cache {record['cache']['path']} --out-root {record['out_root']} --dev-build "
                f"{record['dev_build']} --suites {','.join(record['suites'])}{cases}"
                f"{option_arguments(record['baseline']['options'])})."]
    lines = [f"Null band: {band['path']} (baseline screens of init seeds {', '.join(map(str, band['init_seeds']))})."]
    if band.get("code_sha256") != code:
        measured = (band.get("code_sha256") or "unrecorded")[:12]
        lines.append(f"Warning: the null band was measured under other AlignAIR code ({measured}) than this "
                     f"comparison ({(code or 'unknown')[:12]}): the noise it bounds may have changed with the code.")
    return lines


def _footer(report: Path, layout, key: str, baseline_key: str, times: dict, record: dict) -> list[str]:
    detail = layout.diagnostics_dir(key) / "<S>" / f"against_{baseline_key}.md"
    return [f"Report: {report}", f"Per-suite detail: {detail}",
            f"Wall time: {_minutes(times['total'])} (baseline diagnosis {_minutes(times['baseline_diagnosis'])}, "
            f"training {_minutes(times['training'])} at {record['candidate']['steps_per_second']} steps/s, "
            f"candidate diagnosis {_minutes(times['candidate_diagnosis'])})."]


def _baseline_headline(record: dict, times: dict) -> list[str]:
    return [f"Baseline screen {record['name']}: {_training_text(record)}, from the record cache "
            f"{record['cache']['path']}.", f"Model: {record['candidate']['model_path']}.", _cache_line(record["cache"]),
            f"Screens of other architectures of this protocol compare with the baseline screen of init seed "
            f"{BASELINE_INIT_SEED}; calibrate compares those of several init seeds.",
            f"Wall time: {_minutes(times['total'])} (training {_minutes(times['training'])} at "
            f"{record['candidate']['steps_per_second']} steps/s)."]
