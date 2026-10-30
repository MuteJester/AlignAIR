"""A screen: an architecture trained with the recipe (``recipe``) from the record cache, then compared by the assay with
a baseline at the same step. A yes or no on a candidate in under half an hour, several at once.

**Baseline screens.** A screen of ``NULL_ARCHITECTURE`` (lab-null-v1: conv-v1 under a lab name) compared with nothing
is a baseline screen: one replicate of the baseline, of its ``init_seed``. A screen's *protocol* is what a baseline
must share with it (``protocol``): the recipe id, the record cache's content identity (``cache_identity``), the steps
and the schedule. A screen of another architecture compares with the *registered baseline screen* of its protocol:
the finished screen of lab-null-v1, of the baseline's options (``baseline_options``) and of init seed
``BASELINE_INIT_SEED`` (0) in the screens directory (``registered_baselines``), or with the baseline screen it names
(``against_screen``). Several baseline screens of other init seeds measure the run-to-run noise (``calibrate``), whose
null band (``null_band``) the candidate's headline weighs its Δs against. A screen compares with a milestone of the
baseline's own run instead when told one (``against_milestone``, Tier 2): the milestone's run kept a constant rate.

By default a screen cools its learning rate down (``recipe.schedule``: cosine to a hundredth of it over its steps), so
that replicate runs end close to each other, and draws its initial weights from init seed 0 (``TrainingConfig
.init_seed``): every screen's batches are then the same, whatever its architecture draws when its model is built.

``run_screen(name, steps=..., ...)``:

1. **Refuses** before anything is written (FileNotFoundError, FileExistsError or ValueError, ConfigurationError
   among them): a name that is not one path component; an unknown schedule or an init seed that is not a uint32; a
   baseline that is not the recipe's at that step under that schedule, or that continued another checkpoint, clipped
   its gradients or ran on another GenAIRR than the cache's (``check_baseline``, ``same_genairr``: a baseline is
   trusted by its checkpoint's contents, not its file name or its screen's record): a milestone (``milestone``), a
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
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from alignair.errors import ConfigurationError
from alignair.train.trainer import RECIPE
from alignair_lab.headline import HEADLINE, render_headline
from alignair_lab.noise import decide
from alignair_lab.recipe import (
    BASELINE_ARCHITECTURE,
    BATCH_SIZE,
    DATACONFIGS,
    POLICY,
    SEED,
    recipe_train_args,
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
NULL_ARCHITECTURE = "lab-null-v1"         # ``architectures.null``: its screens are the baseline screens
BASELINE_INIT_SEED = 0                    # the init seed of the registered baseline screen
CHECKPOINT = Path("train") / "model.alignair"   # a screen's trained model, under its directory
SCREEN_RECORD = "screen.json"
NULL_BAND = "null_band.json"              # a calibration's null band (``calibrate``), under its directory
NULL_BAND_KIND = "alignair_lab.null_band"
NULL_BAND_VERSION = 1
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MILESTONE = re.compile(r"model\.step(\d+)\.alignair")
PROGRESS_LINES = 20                       # progress lines printed over a training run
# The width fields of a model configuration, recorded with a baseline's provenance; every configuration field,
# these among them, is compared with the recipe's.
CAPACITY_FIELDS = ("embed_dim", "filters", "block_out", "latent_size_factor", "v_allele_latent_size",
                   "d_allele_latent_size", "j_allele_latent_size", "cls_scale_multiplier", "cls_dim_min", "cls_dim_max",
                   "cls_filter_cap")
_V3_HINT = ("Its training records were drawn otherwise (other seeds, no selection, the uniform indel form), so it is "
            "no baseline for a candidate trained from a v3 record cache: screen against the milestones of a conv-v1 "
            "baseline trained with recipe v3 (--against-milestone DIR), or against a baseline screen.")


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


def _refuse(path, what: str, found: Mapping, expected: Mapping, differing: list[str], **options):
    raise ConfigurationError(f"{path} {what}: " + "; ".join(f"{field} {found.get(field)!r}, not {expected[field]!r}"
                                                            for field in differing),
                             context={"field": differing[0], "fields": differing, "path": str(path)}, **options)


def check_baseline(path: Path, steps: int, *, schedule=None, options: Mapping | None = None,
                   architecture: str = BASELINE_ARCHITECTURE, init_seed: int | None = None) -> dict:
    """The provenance of the baseline checkpoint ``path``, trusted by its contents, not its file name.

    ConfigurationError, naming each difference, unless its header says it was saved at ``steps``; its ``train_args``
    hold the recipe's (``recipe.recipe_train_args`` under ``schedule``: recipe id, optimization and schedule, stream
    policy and spaces; a checkpoint without a ``recipe`` predates recipe v3 and is refused with a hint), say it
    neither clipped its gradients nor continued another checkpoint (a resume draws its later records from other seed
    blocks, a warm start other initial weights; ``started_from``, or ``source_checkpoint_sha256`` where a checkpoint
    predates it) and, given ``init_seed``, that it drew its initial weights from it; it was trained on the recipe's
    references (their DataConfig ``schema_sha256``); and its configuration is, field for field, ``architecture``'s
    with the recipe's default options, or with ``options`` when they are given (a baseline of another width, say).
    The provenance: the ``checkpoint``, step, recipe, architecture, schedule, init seed, git commit, GenAIRR version
    and source digest and reference checksums the checkpoint records, when it was saved, the baseline ``options``
    and its capacity (``CAPACITY_FIELDS``)."""
    import dataclasses
    import GenAIRR.data as gd
    import alignair_lab.architectures  # noqa: F401  (a baseline screen's model is a candidate's)
    from alignair.core import prepare_model
    from alignair.model_file import read_metadata, read_model_state
    from alignair.reference.reference_set import ReferenceSet
    metadata = read_metadata(path)
    training = metadata.get("training") or {}
    if training.get("steps") != steps:
        raise ConfigurationError(f"{path} was saved at step {training.get('steps')}, not at step {steps}",
                                 context={"field": "step", "path": str(path)},
                                 hint="A baseline is trusted by the step its checkpoint was saved at, not by its "
                                      "file name.")
    train_args = training.get("train_args") or {}
    if "recipe" not in train_args:
        raise ConfigurationError(f"{path} was trained before recipe v3: its train_args record no recipe",
                                 context={"field": "recipe", "path": str(path)}, hint=_V3_HINT)
    expected = recipe_train_args(schedule)
    differing = _differences(train_args, expected, expected)
    if differing:
        milestone_rate = "schedule" in differing and schedule is not None and architecture == BASELINE_ARCHITECTURE
        _refuse(path, "was trained with another recipe", train_args, expected, differing,
                hint=("The milestones of the recipe's runs keep a constant rate: screen against them with --schedule "
                      "constant." if milestone_rate else None))
    if train_args.get("grad_clip") is not None:
        raise ConfigurationError(f"{path} was trained with gradient clipping (grad_clip {train_args['grad_clip']!r}); "
                                 "the recipe clips none", context={"field": "grad_clip", "path": str(path)})
    started, source = train_args.get("started_from"), train_args.get("source_checkpoint_sha256")
    if started is not None or source is not None:
        how = ", ".join([*([started] if started else []),
                         *([f"source_checkpoint_sha256 {source[:12]}"] if source else [])])
        raise ConfigurationError(f"{path} continued another checkpoint ({how}): the recipe's baseline trains from "
                                 "its own initial weights on the record stream from step 0",
                                 context={"field": "started_from", "path": str(path)})
    if init_seed is not None and train_args.get("init_seed") != init_seed:
        _refuse(path, "was not trained from its screen's init seed", train_args, {"init_seed": init_seed},
                ["init_seed"])
    sources = [getattr(gd, name) for name in DATACONFIGS]
    references = {"dataconfigs": [entry.get("schema_sha256") for entry in
                                  (metadata.get("reference") or {}).get("dataconfigs", [])]}
    wanted = {"dataconfigs": [getattr(source, "schema_sha256", None) for source in sources]}
    if references != wanted:
        _refuse(path, "was trained on other references", references, wanted, ["dataconfigs"])
    options = dict(options or {})
    if "architecture" in options:
        raise ConfigurationError(f"the baseline is {architecture}: its options cannot name another architecture",
                                 context={"field": "baseline_options"})
    config = read_model_state(path).config
    named = {} if architecture == BASELINE_ARCHITECTURE else {"architecture": architecture}
    recipe_config = prepare_model(ReferenceSet.from_dataconfigs(*sources), **named, **options).config
    fields = [field.name for field in dataclasses.fields(recipe_config)]
    found = {field: getattr(config, field) for field in fields}
    expected = {field: getattr(recipe_config, field) for field in fields}
    differing = _differences(found, expected, fields)
    if differing:
        _refuse(path, f"is not the recipe's {architecture}" + (f" with {options}" if options else ""),
                found, expected, differing,
                hint="A baseline of other options than its architecture's defaults is compared when the screen is "
                     "told them (--baseline-model-option KEY=VALUE).")
    return {"checkpoint": str(path), "step": steps, "recipe": train_args["recipe"],
            "architecture": config.architecture, "schedule": train_args.get("schedule"),
            "init_seed": train_args.get("init_seed"),
            "git_commit": (metadata.get("provenance") or {}).get("git_commit"),
            "genairr_version": metadata.get("genairr_version"),
            "genairr_source_sha256": train_args.get("genairr_source_sha256"),
            "reference_schema_sha256": references["dataconfigs"], "saved": metadata.get("created"),
            "options": options, "capacity": {field: getattr(config, field) for field in CAPACITY_FIELDS}}


def baseline_milestone(milestones: Path, steps: int, options: Mapping | None = None, *, schedule=None) -> dict:
    """The baseline's milestone at ``steps`` in ``milestones`` (``milestone``), checked (``check_baseline``: of
    conv-v1, under ``schedule``); its provenance, of kind ``milestone``."""
    return {"kind": "milestone",
            **check_baseline(milestone(Path(milestones), steps), steps, schedule=schedule, options=options)}


def same_genairr(baseline: Mapping, manifest: Mapping) -> None:
    """Refuse (ConfigurationError) a baseline trained on another GenAIRR than the record cache's: by the GenAIRR
    source digest its checkpoint records, or, for a checkpoint that predates it, by the version this process's
    GenAIRR reports (``GenAIRR.__version__``, as checkpoints record it), which is the cache's: a cache is replayed
    only by the GenAIRR it was written with (``records.CachedRecordSource``)."""
    import GenAIRR
    digest, cached = baseline.get("genairr_source_sha256"), manifest["genairr"]["genairr_source_sha256"]
    if digest is not None:
        if digest != cached:
            raise ConfigurationError(f"the baseline {baseline['checkpoint']} and the record cache come from different "
                                     f"GenAIRR sources: genairr_source_sha256 {digest[:12]}, not {cached[:12]} (the "
                                     "record cache's)", context={"field": "genairr_source_sha256",
                                                                 "path": baseline["checkpoint"]},
                                     hint="Compare a candidate trained from this cache with a baseline trained under "
                                          "the same GenAIRR.")
        return
    version = getattr(GenAIRR, "__version__", None)
    if baseline.get("genairr_version") != version:
        raise ConfigurationError(f"the baseline {baseline['checkpoint']} and the record cache come from different "
                                 f"GenAIRR versions: genairr_version {baseline.get('genairr_version')!r}, not "
                                 f"{version!r} (the GenAIRR this process imports, which wrote the record cache)",
                                 context={"field": "genairr_version", "path": baseline["checkpoint"]},
                                 hint="Compare a candidate trained from this cache with a baseline trained under the "
                                      "same GenAIRR.")


def cache_identity(manifest: Mapping) -> str:
    """A record cache's content identity: the SHA-256 of each stream's seed and its chunks' SHA-256s, in the
    manifest's order. Two caches of one identity hold the same bytes, wherever they are."""
    streams = [[entry["seed"], [chunk["sha256"] for chunk in entry["chunks"]]] for entry in manifest["components"]]
    return hashlib.sha256(json.dumps(streams, separators=(",", ":")).encode()).hexdigest()


def protocol(manifest: Mapping, steps: int, schedule) -> dict:
    """What a baseline screen shares with the screens that compare with it, as ``screen.json`` records it: the recipe
    id, the record cache's identity, the steps and the schedule (as ``train_args`` record it; None: constant)."""
    return {"recipe": RECIPE, "cache": cache_identity(manifest), "steps": steps,
            "schedule": None if schedule is None else schedule.to_dict()}


def _canonical(value):
    return json.loads(json.dumps(value))


def _screen_records(out_root: Path):
    """``(directory, record)`` of every screen under ``out_root`` whose ``screen.json`` reads as a mapping."""
    root = Path(out_root)
    for path in sorted(root.glob(f"*/{SCREEN_RECORD}")) if root.is_dir() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict):
            yield path.parent, record


def baseline_screens(out_root: Path, wanted: Mapping, options: Mapping, init_seed: int) -> list[tuple[Path, dict]]:
    """The screens of ``NULL_ARCHITECTURE`` under ``out_root`` of the protocol ``wanted``, the model options
    ``options`` and ``init_seed``, finished or not: ``(directory, record)``."""
    wanted, options = _canonical(wanted), _canonical(dict(options))
    return [(directory, record) for directory, record in _screen_records(out_root)
            if record.get("architecture") == NULL_ARCHITECTURE and record.get("protocol") == wanted
            and record.get("model_options") == options and record.get("init_seed") == init_seed]


def screen_name(cache, steps: int, schedule: str, init_seed: int) -> str:
    """The name ``calibrate`` gives the baseline screen of ``init_seed``, and the error without one suggests."""
    return f"baseline-{Path(cache).name}-s{steps}-{schedule}-init{init_seed}"


def _option_arguments(options: Mapping) -> str:
    return "".join(f" --model-option {key}={json.dumps(value)}" for key, value in options.items())


def _making_hint(cache, out_root, steps: int, schedule: str, options: Mapping) -> str:
    where = f"--cache {cache} --out-root {out_root}{_option_arguments(options)}"
    return (f"Make one: python -m alignair_lab screen --name {screen_name(cache, steps, schedule, BASELINE_INIT_SEED)} "
            f"--architecture {NULL_ARCHITECTURE} --init-seed {BASELINE_INIT_SEED} --steps {steps} --schedule "
            f"{schedule} {where}; or calibrate, which trains it beside the baseline screens of init seeds 1 and 2 and "
            f"computes the null band: python -m alignair_lab calibrate --steps {steps} --init-seeds 0,1,2 --schedule "
            f"{schedule} {where}")


def baseline_screen(out_root: Path, wanted: Mapping, options: Mapping, *, name: str | None = None, cache=None,
                    schedule: str = DEFAULT_SCHEDULE) -> dict:
    """The baseline screen a screen of the protocol ``wanted`` compares with, and its checkpoint's provenance (kind
    ``screen``, its ``name``, ``screen_dir`` and ``init_seed``, and ``check_baseline``'s).

    ``name``: the screen of that name under ``out_root``, which must be a finished screen of ``NULL_ARCHITECTURE``
    with the model options ``options`` and the protocol ``wanted`` (FileNotFoundError or ConfigurationError
    otherwise). None: the registered baseline screen, the one finished screen of ``baseline_screens`` of init seed
    ``BASELINE_INIT_SEED``; ConfigurationError for none, saying the commands that make one (with ``cache`` and
    ``schedule``, the schedule's name), or for several, naming them."""
    wanted, options = _canonical(wanted), _canonical(dict(options))
    if name is not None:
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError(f"no screen is named {name!r}")
        directory = Path(out_root) / name
        try:
            record = json.loads((directory / SCREEN_RECORD).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"no screen named {name} in {out_root}") from None
        if record.get("architecture") != NULL_ARCHITECTURE:
            raise ConfigurationError(f"the screen {name} is not a baseline screen: architecture "
                                     f"{record.get('architecture')!r}, not {NULL_ARCHITECTURE!r}",
                                     context={"field": "against_screen", "path": str(directory)})
        if record.get("status") != "done":
            raise ConfigurationError(f"the baseline screen {name} has not finished (status {record.get('status')!r})",
                                     context={"field": "against_screen", "path": str(directory)})
        found = {**(record.get("protocol") or {}), "model_options": record.get("model_options")}
        expected = {**wanted, "model_options": options}
        differing = _differences(found, expected, expected)
        if differing:
            _refuse(directory, "is a baseline screen of another protocol", found, expected, differing)
    else:
        matches = baseline_screens(out_root, wanted, options, BASELINE_INIT_SEED)
        done = [(directory, record) for directory, record in matches if record.get("status") == "done"]
        if not done:
            unfinished = "".join(f"; {directory.name} has not finished ({record.get('status')})"
                                 for directory, record in matches)
            raise ConfigurationError(f"no baseline screen of this protocol in {out_root}: no finished screen of "
                                     f"{NULL_ARCHITECTURE} with init seed {BASELINE_INIT_SEED}, these model options "
                                     f"and the protocol {json.dumps(wanted)}{unfinished}",
                                     context={"field": "against_screen", "path": str(out_root)},
                                     hint=_making_hint(cache, out_root, wanted["steps"], schedule, options))
        if len(done) > 1:
            raise ConfigurationError(f"{len(done)} baseline screens of this protocol in {out_root}: "
                                     f"{', '.join(directory.name for directory, _ in done)}",
                                     context={"field": "against_screen", "path": str(out_root)},
                                     hint="Name the one to compare with: --against-screen NAME.")
        directory, record = done[0]
    checkpoint = directory / CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"the baseline screen {directory.name} holds no model: {checkpoint}")
    provenance = check_baseline(checkpoint, wanted["steps"], schedule=_schedule_of(wanted), options=options,
                                architecture=NULL_ARCHITECTURE, init_seed=record.get("init_seed"))
    return {"kind": "screen", "name": directory.name, "screen_dir": str(directory),
            "init_seed": record.get("init_seed"), **provenance}


def _schedule_of(wanted: Mapping):
    from alignair.train import CosineSchedule
    found = wanted["schedule"]
    return None if found is None else CosineSchedule(total_steps=found["total_steps"], min_lr=found["min_lr"])


def null_band(out_root: Path, wanted: Mapping, options: Mapping, suites: Sequence[str],
              max_cases: int | None) -> dict | None:
    """The null band of the protocol ``wanted`` under ``out_root`` (``calibrate``'s ``<dir>/null_band.json``): of
    baseline screens of the model options ``options``, compared on ``suites`` and ``max_cases``. Its ``path``,
    ``init_seeds`` and ``band``; None when there is none; ConfigurationError, naming them, for several."""
    wanted, options = _canonical(wanted), _canonical(dict(options))
    found = []
    root = Path(out_root)
    for path in sorted(root.glob(f"*/{NULL_BAND}")) if root.is_dir() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(record, dict) and record.get("kind") == NULL_BAND_KIND
                and record.get("version") == NULL_BAND_VERSION and record.get("protocol") == wanted
                and record.get("options") == options and record.get("suites") == list(suites)
                and record.get("max_cases") == max_cases):
            found.append((path, record))
    if len(found) > 1:
        raise ConfigurationError(f"{len(found)} null bands of this protocol in {out_root}: "
                                 f"{', '.join(str(path) for path, _ in found)}",
                                 context={"field": "null_band", "path": str(out_root)},
                                 hint="Keep one calibration of a protocol: move the others away.")
    if not found:
        return None
    path, record = found[0]
    return {"path": str(path), "init_seeds": record["init_seeds"], "band": record["band"]}


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


def check_init_seed(init_seed) -> int:
    """``init_seed``, a uint32 (ValueError otherwise)."""
    if isinstance(init_seed, bool) or not isinstance(init_seed, int) or not 0 <= init_seed < 2**32:
        raise ValueError(f"an init seed is an integer from 0 to 2**32 - 1, got {init_seed!r}")
    return init_seed


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
    init_seed = check_init_seed(init_seed)
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
    names = band = None
    if provenance is not None:
        same_genairr(provenance, manifest)
        names = built_suites(Path(dev_build), suites)
        band = null_band(Path(out_root), wanted, baseline_options, names, max_cases)
    config = training_config(steps=steps, architecture=architecture, model_options=model_options, device=device,
                             data_workers=data_workers, schedule=learning, init_seed=init_seed)
    _prepare_candidate(config)
    screen = Path(out_root) / name
    Path(out_root).mkdir(parents=True, exist_ok=True)
    try:
        screen.mkdir()
    except FileExistsError:
        raise FileExistsError(f"a screen named {name!r} exists ({screen}); choose another --name") from None

    record = {"name": name, "status": "preparing", "started": _now(), "role": role, "steps": steps,
              "architecture": architecture, "model_options": model_options, "init_seed": init_seed,
              "schedule": schedule, "protocol": wanted, "suites": names, "variants": list(variants),
              "max_cases": max_cases, "device": device, "data_workers": data_workers, "eval_batch_size": eval_batch_size,
              "recipe": {"id": RECIPE, "dataconfigs": list(DATACONFIGS), "batch_size": BATCH_SIZE, "lr": config.lr,
                         "seed": SEED},
              "cache": {"path": source.path, "identity": wanted["cache"], "steps": manifest["steps"],
                        "records": manifest["records"],
                        "genairr_source_sha256": manifest["genairr"]["genairr_source_sha256"],
                        "genairr_version": manifest["genairr"].get("genairr_version"), "recipe": manifest["recipe"],
                        "code_sha256": manifest["code"]["sha256"], "created": manifest["created"],
                        "verification": source.verification},
              "baseline": provenance, "null_band": band,
              "dev_build": str(Path(dev_build).resolve()), "screen_dir": str(screen.resolve())}

    def stage(status: str, **fields):
        record.update(status=status, **fields)
        _write_json(screen / SCREEN_RECORD, record)
        echo(f"[screen {name}] {status}")

    try:
        times = {}
        if provenance is not None:
            layout = _assay_layout(screen / "assay", Path(dev_build), names)
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
            headline = render_headline(summary, title=_title(record, result, base.model_key), null_band=limits)
            headline += "\n\n" + "\n".join(_footer(report, layout, result.model_key, base.model_key, times, record))
            record["candidate"]["model_key"] = result.model_key
            record.update(report=str(report), comparison=summary,
                          decisions=decide(summary, limits, [metric for metric, _ in HEADLINE]))
        (screen / "headline.txt").write_text(headline + "\n", encoding="utf-8")
        record.update(seconds={key: round(value, 1) for key, value in times.items()}, finished=_now())
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


def _training_text(record: dict) -> str:
    options = f" with {json.dumps(record['model_options'])}" if record["model_options"] else ""
    return (f"{record['architecture']}{options}, {record['steps']} steps ({record['schedule']} schedule, init seed "
            f"{record['init_seed']})")


def _title(record: dict, result, baseline_key: str) -> list[str]:
    baseline = record["baseline"]
    against = (f"the baseline screen {baseline['name']} ({NULL_ARCHITECTURE}, init seed {baseline['init_seed']})"
               if baseline["kind"] == "screen" else f"the baseline {BASELINE_ARCHITECTURE}'s milestone")
    return [f"Screen {record['name']}: {_training_text(record)}, against {against} at step {record['steps']}.",
            f"Candidate {result.model_key} ({record['candidate']['model_path']}); baseline {baseline_key} "
            f"({baseline['checkpoint']}).", _provenance_line(baseline), _cache_line(record["cache"]),
            _band_line(record), ""]


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


def _band_line(record: dict) -> str:
    band = record["null_band"]
    if band is not None:
        return (f"Null band: {band['path']} (baseline screens of init seeds "
                f"{', '.join(map(str, band['init_seeds']))}).")
    cases = "" if record["max_cases"] is None else f" --max-cases {record['max_cases']}"
    return (f"Null band: none of this protocol, these suites and cases, so uncalibrated (python -m alignair_lab "
            f"calibrate --steps {record['steps']} --init-seeds 0,1,2 --schedule {record['schedule']} --suites "
            f"{','.join(record['suites'])}{cases}).")


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
