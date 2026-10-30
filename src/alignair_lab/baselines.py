"""Baselines: what a screen is compared with, and what it must share with it. A baseline is trusted by the contents of
its checkpoint, never by its file name or its screen's record.

- ``check_baseline``: a baseline checkpoint of the recipe at a step, under a schedule, of an architecture and options:
  neither gradient-clipped nor continued from another checkpoint; ``baseline_milestone`` checks the milestone of the
  baseline's own run at a step (``milestone``).
- ``same_genairr``: a baseline trained on the record cache's GenAIRR.
- ``protocol``: what a baseline screen shares with the screens that compare with it: the recipe id, the record cache's
  content identity (``cache_identity``), the steps and the schedule.
- ``baseline_screens`` and ``baseline_screen``: the screens of ``NULL_ARCHITECTURE`` of a protocol, and the one a screen
  compares with (the registered baseline screen, of init seed ``BASELINE_INIT_SEED``, or one named).
- ``null_band``: the null band of a protocol, written by ``calibrate``.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from alignair.errors import ConfigurationError
from alignair.train.trainer import RECIPE
from alignair_lab.recipe import BASELINE_ARCHITECTURE, DATACONFIGS, recipe_train_args

NULL_ARCHITECTURE = "lab-null-v1"         # ``architectures.null``: its screens are the baseline screens
BASELINE_INIT_SEED = 0                    # the init seed of the registered baseline screen
CHECKPOINT = Path("train") / "model.alignair"   # a screen's trained model, under its directory
SCREEN_RECORD = "screen.json"
NULL_BAND = "null_band.json"              # a calibration's null band (``calibrate``), under its directory
NULL_BAND_KIND = "alignair_lab.null_band"
NULL_BAND_VERSION = 1
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MILESTONE = re.compile(r"model\.step(\d+)\.alignair")
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
                    schedule: str) -> dict:
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


def check_init_seed(init_seed) -> int:
    """``init_seed``, a uint32 (ValueError otherwise)."""
    if isinstance(init_seed, bool) or not isinstance(init_seed, int) or not 0 <= init_seed < 2**32:
        raise ValueError(f"an init seed is an integer from 0 to 2**32 - 1, got {init_seed!r}")
    return init_seed


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)
