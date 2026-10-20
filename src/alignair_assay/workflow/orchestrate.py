"""``all``: build the suites, run the oracle and two tools, score, compare and write the run report, resumably.

A run directory belongs to one configuration (``RunConfig``), recorded in ``manifest.json`` with its
``config_hash``. ``run_all`` refuses a directory made for another configuration, naming the fields that differ,
unless it is forced; then it redoes only what the change affects, because every step is skipped only when its own
record shows it is current: a suite whose ``build.json`` has the suite's config hash, a run whose ``run.json``
records success on the suite's current files with the tool's current settings. Scores, comparisons and the report
are always recomputed from the current runs, so they never mix in stale outputs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import genairr_provenance, git_state, sha256_file
from alignair_assay.core.reference import DEFAULT_REFERENCE, Reference
from alignair_assay.report import run_report_markdown
from alignair_assay.runners import RUNNERS, make_runner
from alignair_assay.runners.alignair import DEFAULT_IMAGE, DEFAULT_MODEL
from alignair_assay.runners.arms import run_genotype_arm
from alignair_assay.runners.base import RunnerOptions, RunResult, absolute_path, read_run_json, run_suite
from alignair_assay.runners.reference import ensure_shared_reference
from alignair_assay.scoring.pipeline import load_truth, score_tool, write_scores
from alignair_assay.suites import get_suite
from alignair_assay.suites.engine import DEFAULT_SEED, check_split
from alignair_assay.truth.io import TRUTH_VERSION, read_truth_header
from alignair_assay.workflow.build import build_suite
from alignair_assay.workflow.build import config_hash as suite_config_hash
from alignair_assay.workflow.scoring import ToolScores, comparison_report, score_report, write_report
from alignair_assay.workflow.summary import (
    ALPHA,
    ORACLE,
    collect_run_summary,
    expected_runs,
    inputs_unchanged,
    read_json,
    relative,
    run_status,
)

MANIFEST_KIND = "alignair_assay.manifest"
MANIFEST_VERSION = "1"
# Per tool, the settings that change what it predicts. They are part of the run directory's configuration, and a
# completed run made with other values is redone. A tool not listed (the oracle, test fakes) has none.
PREDICTION_SETTINGS = {"igblast": ("igblast_dir",), "alignair": ("alignair_image", "alignair_model"),
                       "alignair-local": ("alignair_checkpoint", "alignair_checkpoint_sha256", "device")}


@dataclass(frozen=True)
class RunConfig:
    """One ``all`` run: the ``suites`` to build and score, the two compared ``tools`` (a, then b; the oracle always
    runs once per suite as well), the ``split``, truth ``reference``, ``seed`` and ``scale`` the suites are built
    with, the ``workers`` building them, the tool ``threads``, the tool settings (``igblast_dir``, the AlignAIR
    image, registry model and local checkpoint, the torch ``device``) and whether to measure ``speed``."""

    suites: tuple[str, ...]
    tools: tuple[str, ...]
    split: str = "dev"
    reference: str = DEFAULT_REFERENCE
    seed: int = DEFAULT_SEED
    scale: float = 1.0
    workers: int = 8
    threads: int = 8
    igblast_dir: str | None = None
    alignair_image: str | None = None
    alignair_model: str = DEFAULT_MODEL
    alignair_checkpoint: str | None = None
    device: str = "cpu"
    speed: bool = False


def tool_settings(cfg: RunConfig, tool: str) -> dict:
    """The settings ``tool`` predicts with under ``cfg`` (its ``PREDICTION_SETTINGS``): paths made absolute, the
    default image filled in, and for a local checkpoint its SHA-256 (None for a missing file), so a model retrained
    in place is a new configuration."""
    names = PREDICTION_SETTINGS.get(tool, ())
    paths = RunnerOptions(igblast_dir=cfg.igblast_dir, alignair_checkpoint=cfg.alignair_checkpoint)
    values = {"igblast_dir": paths.igblast_dir, "alignair_image": cfg.alignair_image or DEFAULT_IMAGE,
              "alignair_model": cfg.alignair_model, "alignair_checkpoint": paths.alignair_checkpoint,
              "device": cfg.device}
    if "alignair_checkpoint_sha256" in names:
        checkpoint = paths.alignair_checkpoint
        values["alignair_checkpoint_sha256"] = (sha256_file(checkpoint) if checkpoint and Path(checkpoint).is_file()
                                                else None)
    return {name: values[name] for name in names}


def identity(cfg: RunConfig) -> dict:
    """The configuration a run directory's outputs depend on: ``suites``, ``tools``, ``split``, ``reference``,
    ``seed``, ``scale`` and each tool's ``tool_settings``. Execution settings (``workers``, ``threads``,
    ``speed``) are left out: they change neither the cases nor the predictions."""
    fields = {"suites": list(cfg.suites), "tools": list(cfg.tools), "split": cfg.split, "reference": cfg.reference,
              "seed": int(cfg.seed), "scale": float(cfg.scale)}
    for tool in cfg.tools:
        fields.update(tool_settings(cfg, tool))
    return fields


def config_hash(cfg: RunConfig) -> str:
    """Hex SHA-256 of the canonical JSON (sorted keys, no spaces) of ``identity(cfg)``."""
    return _digest(identity(cfg))


def check_config(cfg: RunConfig) -> None:
    """ValueError for a configuration ``run_all`` cannot run, before anything is written: an unknown suite, split
    or tool, not exactly two different tools, a scale that is not positive, fewer than one worker or thread, or
    ``speed`` (the speed protocol is not available yet)."""
    if not cfg.suites:
        raise ValueError("no suite to run")
    for name in cfg.suites:
        get_suite(name)
    if len(cfg.tools) != 2 or cfg.tools[0] == cfg.tools[1]:
        raise ValueError(f"all compares exactly two tools, a and b; got {list(cfg.tools)}")
    for tool in cfg.tools:
        if tool not in RUNNERS:
            raise ValueError(f"unknown tool {tool!r}; known tools: {', '.join(RUNNERS)}")
    check_split(cfg.split)
    if not cfg.scale > 0:
        raise ValueError(f"scale must be positive, got {cfg.scale}")
    if cfg.workers < 1 or cfg.threads < 1:
        raise ValueError(f"workers and threads must be at least 1, got {cfg.workers} and {cfg.threads}")
    if cfg.speed:
        raise ValueError("speed measurement (--speed) is not available yet")


def run_all(cfg: RunConfig, layout: RunLayout, *, force: bool = False) -> int:
    """Build, run, score, compare and report ``cfg`` in the run directory ``layout``; returns 0, or 2 when any run
    failed or the oracle is not perfect on a suite (the report names which).

    1. Manifest: ``check_config``, then write ``manifest.json``, or check the existing one. One made for another
       configuration (``config_hash``) raises ValueError naming the fields that differ, before anything changes,
       unless ``force``; forced, the manifest is replaced (``replaced`` records the old hash and the changed
       fields) and the steps below redo what the change affects.
    2. Build each suite (``workflow.build.build_suite``), skipped when its ``build.json`` has the suite's config
       hash and its truth file the current truth version.
    3. Shared reference: with ``alignair`` among the tools and no shared reference yet, the image exports its
       model's reference there; then ``ensure_shared_reference`` verifies it (or writes it from the truth), so a
       model with another reference stops the run before any tool runs.
    4. Runs (``summary.expected_runs``): the oracle once per suite, then each tool on each suite and arm, through
       ``run_suite`` or ``run_genotype_arm``. A run is skipped when ``run_is_current``. Every tool with a run to
       make is prepared before the first run.
    5. Scoring, comparison and the report: ``score_and_report``.
    """
    check_config(cfg)
    reference = Reference.from_dataconfig(cfg.reference)
    layout = RunLayout(absolute_path(layout.root))
    _manifest(cfg, layout, force=force)
    layout.report_json.unlink(missing_ok=True)
    layout.report_md.unlink(missing_ok=True)
    _build(cfg, layout, reference)
    options = RunnerOptions(igblast_dir=cfg.igblast_dir, threads=cfg.threads, alignair_image=cfg.alignair_image,
                            alignair_model=cfg.alignair_model, alignair_checkpoint=cfg.alignair_checkpoint,
                            device=cfg.device, cache_dir=str(layout.work_dir("alignair") / "cache"))
    _shared_reference(cfg, layout, reference, options)
    _run(cfg, layout, reference, options)
    return score_and_report(cfg, layout, reference=reference)


def run_is_current(layout: RunLayout, tool: str, suite: str, arm: str, settings: dict) -> bool:
    """True when the run's ``run.json`` and predictions exist, the record's return code is 0, its recorded
    ``settings`` are ``settings`` and its input files are unchanged (``summary.inputs_unchanged``)."""
    path = layout.run_json(tool, suite, arm)
    if not path.is_file() or not layout.predictions(tool, suite, arm).is_file():
        return False
    try:
        record = read_run_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return (record.returncode == 0 and record.provenance.get("settings") == settings
            and inputs_unchanged(layout, suite, record.provenance))


def score_and_report(cfg: RunConfig, layout: RunLayout, *, reference: Reference | None = None,
                     command: str = "all") -> int:
    """Score, compare and report the runs of ``cfg`` in ``layout``; returns 0, or 2 when a run failed, is
    missing, stale or unscorable, or the oracle is not perfect on a suite.

    For every suite and arm the old ``scores/<S>/<arm>/`` is removed. Each run that passes ``summary.run_status``
    (which applies ``require_successful_run`` and checks the inputs are unchanged) is scored as the ``score``
    command scores it: ``<tool>.scores.jsonl.gz``, ``<tool>.report.json`` and ``<tool>.report.md``, with the
    suite's primary metrics leading the markdown. The oracle is scored on the shared arm. When both tools are
    scored, ``comparison.json`` and ``comparison.md`` are written as the ``compare`` command writes them (Holm at
    ``summary.ALPHA``). Then ``report.json`` (``summary.collect_run_summary``) and ``report.md``
    (``report.run.run_report_markdown``) are written. ``command`` is recorded in the reports' provenance.
    """
    reference = reference or Reference.from_dataconfig(cfg.reference)
    errors = _score(cfg, layout, reference, command=command)
    summary = collect_run_summary(layout, cfg, scoring_errors=errors)
    layout.report_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    layout.report_md.write_text(run_report_markdown(summary), encoding="utf-8")
    print(f"report: {layout.report_md}", flush=True)
    for run in summary["failed_runs"]:
        donor = f" on donor {run['failed_donor']}" if run.get("failed_donor") else ""
        print(f"error: {run['tool']} on {run['suite']}/{run['arm']} {run['status']}{donor}: {run.get('error')}",
              file=sys.stderr)
    for failure in summary["oracle_failures"]:
        print(f"error: the oracle is {failure['status']} on {failure['suite']} ({failure['problems']} problems): "
              "the suite's truth has a defect", file=sys.stderr)
    return 2 if summary["failed_runs"] or summary["oracle_failures"] else 0


def read_run_config(layout: RunLayout) -> RunConfig:
    """The ``RunConfig`` recorded in the run directory's ``manifest.json``. FileNotFoundError without one (the
    directory was not made by ``all``)."""
    manifest = read_json(layout.manifest)
    if manifest is None:
        raise FileNotFoundError(f"no readable {layout.manifest}: the run directory was not made by `all`")
    names = {field.name for field in dataclasses.fields(RunConfig)}
    config = {key: value for key, value in manifest["config"].items() if key in names}
    return RunConfig(**{**config, "suites": tuple(config["suites"]), "tools": tuple(config["tools"])})


def _digest(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _manifest(cfg: RunConfig, layout: RunLayout, *, force: bool) -> None:
    """Write the manifest, keep an identical one, or refuse (ValueError) one made for another configuration."""
    current = identity(cfg)
    digest = _digest(current)
    replaced = None
    if layout.manifest.is_file():
        old = read_json(layout.manifest) or {}
        if old.get("config_hash") == digest:
            return
        previous = old.get("identity") or {}
        changed = [name for name in sorted(previous.keys() | current.keys()) if previous.get(name) != current.get(name)]
        if not force:
            differences = "; ".join(f"{name} {previous.get(name)} -> {current.get(name)}" for name in changed)
            raise ValueError(f"{layout.manifest} was written for another configuration ({differences}). Rerun with "
                             "that configuration, choose another --run-dir, or pass --force to rebuild and re-run "
                             "what the change affects")
        replaced = {"config_hash": old.get("config_hash"), "changed": changed}
    manifest = {
        "kind": MANIFEST_KIND, "version": MANIFEST_VERSION, "config": dataclasses.asdict(cfg), "identity": current,
        "config_hash": digest, "split": cfg.split,
        "host": {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
                 "logical_cpus": os.cpu_count()},
        "git": git_state(Path(__file__).resolve().parent), "genairr": genairr_provenance(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "replaced": replaced,
    }
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _build(cfg: RunConfig, layout: RunLayout, reference: Reference) -> None:
    digest = reference.digest()
    for name in cfg.suites:
        wanted = suite_config_hash(name, split=cfg.split, reference_digest=digest, seed=int(cfg.seed),
                                   scale=float(cfg.scale))
        record = read_json(layout.build_json(name)) or {}
        if record.get("config_hash") == wanted and _truth_version(layout, name) == TRUTH_VERSION:
            print(f"build {name}: up to date, skipped", flush=True)
            continue
        build = build_suite(name, layout, reference_name=cfg.reference, split=cfg.split, seed=cfg.seed,
                            scale=cfg.scale, workers=cfg.workers)
        print(f"build {name}: {build['n_cases']} cases in {build['seconds']:.1f} s", flush=True)


def _truth_version(layout: RunLayout, suite: str) -> str | None:
    try:
        return (read_truth_header(layout.truth(suite)) or {}).get("version")
    except (OSError, ValueError):
        return None


def _shared_reference(cfg: RunConfig, layout: RunLayout, reference: Reference, options: RunnerOptions) -> None:
    """Export AlignAIR's reference as the shared one when it is a tool and there is none yet, then verify the
    shared reference (or write it from the truth). An exported reference that fails verification is removed, so a
    later run with another model exports again."""
    exported = False
    if "alignair" in cfg.tools and not layout.reference_fasta.is_file():
        layout.reference_dir.mkdir(parents=True, exist_ok=True)
        make_runner("alignair", options).export_reference(layout.reference_fasta)
        exported = True
    try:
        ensure_shared_reference(layout, reference)
    except ValueError:
        if exported:
            layout.reference_fasta.unlink(missing_ok=True)
        raise
    source = "alignair export" if exported else (read_json(layout.reference_json) or {}).get("source")
    print(f"reference: {relative(layout, layout.reference_fasta)} is the truth reference (source: {source})",
          flush=True)


def _run(cfg: RunConfig, layout: RunLayout, reference: Reference, options: RunnerOptions) -> None:
    """Make every expected run that is not current; every tool with a run to make is prepared first."""
    jobs = expected_runs(cfg)
    settings = {tool: tool_settings(cfg, tool) for tool in dict.fromkeys(tool for tool, _, _ in jobs)}
    pending = []
    for tool, suite, arm in jobs:
        if run_is_current(layout, tool, suite, arm, settings[tool]):
            print(f"run {tool} {suite}/{arm}: up to date, skipped", flush=True)
        else:
            pending.append((tool, suite, arm))
    runners = {}
    for tool in dict.fromkeys(tool for tool, _, _ in pending):
        runner = make_runner(tool, options)
        if runner.name != tool:
            raise ValueError(f"the runner registered as {tool!r} is named {runner.name!r}")
        runner.prepare(layout, reference)
        runners[tool] = runner
    for tool, suite, arm in pending:
        if arm == "genotype":
            result = run_genotype_arm(runners[tool], layout, suite, reference, tool=tool, settings=settings[tool])
        else:
            result = run_suite(runners[tool], layout, suite, arm=arm, settings=settings[tool])
        print(_run_line(result), flush=True)


def _run_line(result: RunResult) -> str:
    rss = "not measured" if result.peak_rss_kb is None else f"{result.peak_rss_kb} KB"
    line = (f"run {result.tool} {result.suite}/{result.arm}: return code {result.returncode}, {result.n_reads} reads, "
            f"{result.wall_seconds:.1f} s, peak RSS {rss}")
    if result.returncode != 0:
        donor = result.provenance.get("failed_donor")
        line += (f" on donor {donor}" if donor else "") + f": {result.provenance.get('error')}"
    return line


def _score(cfg: RunConfig, layout: RunLayout, reference: Reference, *, command: str) -> list[dict]:
    """Write every suite and arm's scores and comparison; returns the runs that passed ``run_status`` but could
    not be scored (``run_status``-shaped, status ``not scored``)."""
    a, b = cfg.tools
    errors = []
    for name in cfg.suites:
        suite = get_suite(name)
        header, truths = load_truth(layout.truth(name), cfg.reference, reference)
        for arm in suite.arms:
            shutil.rmtree(layout.scores_dir(name, arm), ignore_errors=True)
            layout.scores_dir(name, arm).mkdir(parents=True)
            scored = {}
            for tool in dict.fromkeys(([ORACLE] if arm == "shared" else []) + [a, b]):
                status = run_status(layout, tool, name, arm)
                if status["status"] != "ok":
                    continue  # the summary lists it as a failed run
                predictions = layout.predictions(tool, name, arm)
                try:
                    scored[tool] = ToolScores(tool, *score_tool(truths, predictions, reference))
                except (FileNotFoundError, ValueError) as error:
                    errors.append({**status, "status": "not scored", "error": str(error)})
                    continue
                report = score_report(scored[tool], truths, header, command=command,
                                      inputs={"truth": layout.truth(name), "predictions": predictions,
                                              "run": layout.run_json(tool, name, arm)},
                                      reference_name=cfg.reference)
                write_scores(scored[tool].scores, layout.tool_scores(name, arm, tool))
                write_report(report, layout.tool_report_json(name, arm, tool), layout.tool_report_md(name, arm, tool),
                             headline=suite.primary)
            compared = a in scored and b in scored
            if compared:
                report = comparison_report(scored[a], scored[b], truths, header, command=command,
                                           inputs={"truth": layout.truth(name), "a": layout.predictions(a, name, arm),
                                                   "b": layout.predictions(b, name, arm)},
                                           reference_name=cfg.reference, alpha=ALPHA)
                write_report(report, layout.comparison_json(name, arm), layout.comparison_md(name, arm),
                             headline=suite.primary)
            print(f"score {name}/{arm}: scored {', '.join(scored) or 'nothing'}; "
                  + (f"compared {a} with {b} in {relative(layout, layout.comparison_md(name, arm))}" if compared
                     else "not compared"), flush=True)
    return errors
