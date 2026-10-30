"""A screen: a candidate architecture trained with the recipe (``recipe``) from the record cache, then compared by the
assay with the baseline's milestone at the same step. A yes or no on a candidate in under half an hour, several at
once.

``run_screen(name, steps=..., ...)``:

1. **Refuses** before anything is written (FileNotFoundError, FileExistsError or ValueError, ConfigurationError
   among them): a name that is not one path component; a step the baseline has no milestone at, or a milestone that
   is not the recipe's baseline at that step (``baseline_milestone``: it is trusted by its contents, not its file
   name); a cache that cannot feed the steps or does not reproduce the live stream
   (``records.CachedRecordSource.require``); an unknown architecture or model option (the candidate's model setup is
   prepared); a suite not built in the dev build; and a screen name already in use.
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

``<screen>/screen.json`` holds the request, the cache and baseline used (the milestone's step, recipe, git commit and
GenAIRR identity, also on the headline), the status (updated at each stage; a failure is recorded before it is
raised), the model keys, the report path, the times and the comparison summary.
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

from alignair.errors import ConfigurationError
from alignair_lab.headline import render_headline
from alignair_lab.recipe import (
    BASELINE_ARCHITECTURE,
    BATCH_SIZE,
    DATACONFIGS,
    POLICY,
    SEED,
    recipe_train_args,
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
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MILESTONE = re.compile(r"model\.step(\d+)\.alignair")
PROGRESS_LINES = 20                       # progress lines printed over a training run
# The width fields of a model configuration, recorded with a baseline milestone's provenance; every configuration field,
# these among them, is compared with the recipe's conv-v1.
CAPACITY_FIELDS = ("embed_dim", "filters", "block_out", "latent_size_factor", "v_allele_latent_size",
                   "d_allele_latent_size", "j_allele_latent_size", "cls_scale_multiplier", "cls_dim_min", "cls_dim_max",
                   "cls_filter_cap")
_V3_HINT = ("Its training records were drawn otherwise (other seeds, no selection, the uniform indel form), so it is "
            "no baseline for a candidate trained from a v3 record cache: screen against the milestones of a conv-v1 "
            "baseline trained with recipe v3 (--milestones).")


def milestone(milestones: Path, steps: int) -> Path:
    """The baseline's checkpoint at ``steps``; FileNotFoundError, naming the steps it has, when there is none."""
    path = Path(milestones) / f"model.step{steps}.alignair"
    if not path.is_file():
        held = sorted(int(match.group(1)) for match in map(_MILESTONE.fullmatch, os.listdir(milestones))
                      if match) if Path(milestones).is_dir() else []
        raise FileNotFoundError(f"the baseline has no milestone at step {steps}: {milestones} holds steps "
                                f"{', '.join(map(str, held)) or 'none'}")
    return path


def _differences(found: Mapping, expected: Mapping, fields) -> list[str]:
    absent = object()
    return [field for field in fields if found.get(field, absent) != expected[field]]


def _refuse(path: Path, what: str, found: Mapping, expected: Mapping, differing: list[str], **options):
    raise ConfigurationError(f"{path} {what}: " + "; ".join(f"{field} {found.get(field)!r}, not {expected[field]!r}"
                                                            for field in differing),
                             context={"field": differing[0], "fields": differing, "path": str(path)}, **options)


def baseline_milestone(milestones: Path, steps: int, options: Mapping | None = None) -> dict:
    """The baseline's milestone at ``steps``, trusted by its contents, not its file name; its provenance.

    ConfigurationError, naming each difference, unless its header says it was saved at ``steps``, its ``train_args``
    hold the recipe's (``recipe.recipe_train_args``: recipe id, optimization, stream policy and spaces; a checkpoint
    without a ``recipe`` predates recipe v3 and is refused with a hint), it was trained on the recipe's references
    (their DataConfig ``schema_sha256``), and its configuration is, field for field, the recipe's conv-v1 with its
    default options, or with ``options`` when they are given (a baseline of another width, say). The provenance: the
    path, step, recipe, git commit, GenAIRR version and reference checksums the checkpoint records, when it was
    saved, the baseline ``options`` the screen was told and the milestone's capacity (``CAPACITY_FIELDS``)."""
    import dataclasses
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.model_file import read_metadata, read_model_state
    from alignair.reference.reference_set import ReferenceSet
    path = milestone(Path(milestones), steps)
    metadata = read_metadata(path)
    training = metadata.get("training") or {}
    if training.get("steps") != steps:
        raise ConfigurationError(f"{path} was saved at step {training.get('steps')}, not at step {steps}",
                                 context={"field": "step", "path": str(path)},
                                 hint="A milestone is trusted by the step its checkpoint was saved at, not by its "
                                      "file name.")
    train_args = training.get("train_args") or {}
    if "recipe" not in train_args:
        raise ConfigurationError(f"{path} was trained before recipe v3: its train_args record no recipe",
                                 context={"field": "recipe", "path": str(path)}, hint=_V3_HINT)
    expected = recipe_train_args()
    differing = _differences(train_args, expected, expected)
    if differing:
        _refuse(path, "was trained with another recipe", train_args, expected, differing)
    sources = [getattr(gd, name) for name in DATACONFIGS]
    references = {"dataconfigs": [entry.get("schema_sha256") for entry in
                                  (metadata.get("reference") or {}).get("dataconfigs", [])]}
    wanted = {"dataconfigs": [getattr(source, "schema_sha256", None) for source in sources]}
    if references != wanted:
        _refuse(path, "was trained on other references", references, wanted, ["dataconfigs"])
    options = dict(options or {})
    if "architecture" in options:
        raise ConfigurationError(f"the baseline is {BASELINE_ARCHITECTURE}: its options cannot name another "
                                 "architecture", context={"field": "baseline_options"})
    config = read_model_state(path).config
    recipe_config = prepare_model(ReferenceSet.from_dataconfigs(*sources), **options).config
    fields = [field.name for field in dataclasses.fields(recipe_config)]
    found = {field: getattr(config, field) for field in fields}
    expected = {field: getattr(recipe_config, field) for field in fields}
    differing = _differences(found, expected, fields)
    if differing:
        _refuse(path, f"is not the recipe's {BASELINE_ARCHITECTURE}" + (f" with {options}" if options else ""),
                found, expected, differing,
                hint="A baseline of other options than conv-v1's defaults is compared when the screen is told them "
                     "(--baseline-model-option KEY=VALUE).")
    return {"milestone": str(path), "step": steps, "recipe": train_args["recipe"], "architecture": config.architecture,
            "git_commit": (metadata.get("provenance") or {}).get("git_commit"),
            "genairr_version": metadata.get("genairr_version"),
            "reference_schema_sha256": references["dataconfigs"], "saved": metadata.get("created"),
            "options": options, "capacity": {field: getattr(config, field) for field in CAPACITY_FIELDS}}


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
               dev_build=DEV_BUILD, milestones=MILESTONES, out_root=SCREENS, baseline_options: Mapping | None = None,
               echo: Callable[[str], None] | None = None) -> dict:
    """Screen ``architecture`` with ``model_options`` for ``steps`` steps against the baseline milestone at that step
    (see the module docstring); returns the screen record (``screen.json``) with its printed ``headline``.

    The baseline is the recipe's conv-v1 with its default options, or with ``baseline_options`` (recorded in
    ``screen.json`` and on the headline). Its milestone and the record cache must come from one GenAIRR version.
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
    provenance = baseline_milestone(Path(milestones), steps, baseline_options)
    baseline = provenance["milestone"]
    source = CachedRecordSource(cache)
    manifest = source.require(DATACONFIGS, policy=POLICY, seed=SEED, records=steps * BATCH_SIZE)
    version = manifest["genairr"].get("genairr_version")
    if provenance["genairr_version"] != version:
        raise ConfigurationError(f"the baseline milestone {baseline} and the record cache come from different GenAIRR "
                                 f"versions: genairr_version {provenance['genairr_version']!r}, not {version!r} (the "
                                 "record cache's)", context={"field": "genairr_version", "path": baseline},
                                 hint="Compare a candidate trained from this cache with a baseline trained under the "
                                      "same GenAIRR.")
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
                        "genairr_version": version, "recipe": manifest["recipe"],
                        "code_sha256": manifest["code"]["sha256"], "created": manifest["created"],
                        "verification": source.verification},
              "baseline": provenance,
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
            f"({record['baseline']['milestone']}).", _provenance_line(record["baseline"]), _cache_line(record["cache"]),
            ""]


def _provenance_line(baseline: dict) -> str:
    commit = (baseline.get("git_commit") or "unknown")[:12]
    references = ", ".join((schema or "unknown")[:12] for schema in baseline["reference_schema_sha256"])
    options = (f"; baseline options {json.dumps(baseline['options'])}" if baseline.get("options")
               else "; conv-v1's default options")
    return (f"Baseline milestone: saved at step {baseline['step']} by recipe {baseline['recipe']}, at git {commit}, "
            f"with GenAIRR {baseline.get('genairr_version') or 'unknown'} (reference {references}), on "
            f"{baseline.get('saved') or 'an unrecorded date'}{options}.")


def _cache_line(cache: dict) -> str:
    verification = cache.get("verification") or {}
    if verification.get("code_changed"):
        return (f"Record cache: producing code changed since the write; deep-verified {verification['records']} "
                "records live.")
    return (f"Record cache: producing code unchanged since the write; verified {verification.get('records', 0)} "
            "records live.")


def _footer(report: Path, layout, key: str, baseline_key: str, times: dict, record: dict) -> list[str]:
    detail = layout.diagnostics_dir(key) / "<S>" / f"against_{baseline_key}.md"
    return [f"Report: {report}", f"Per-suite detail: {detail}",
            f"Wall time: {_minutes(times['total'])} (baseline diagnosis {_minutes(times['baseline_diagnosis'])}, "
            f"training {_minutes(times['training'])} at {record['candidate']['steps_per_second']} steps/s, "
            f"candidate diagnosis {_minutes(times['candidate_diagnosis'])})."]
