"""``all``: build the suites, run the oracle and two tools, time them (``--speed``), score, compare and write the
run report, resumably.

A run directory belongs to one configuration, recorded in ``manifest.json`` with its ``config_hash``: the
``RunConfig`` fields the outputs depend on plus each tool's fingerprint (``Runner.fingerprint``: binary digests,
the resolved image ID, the model file's digest), so a tool changed in place is a new configuration too.
``run_all`` refuses a directory made for another configuration, naming the fields that differ, unless it is
forced; then it redoes only what the change affects, because every step is skipped only when its own record shows
it is current: a suite whose ``build.json`` has the suite's config hash, a run whose ``run.json`` records success
on the suite's current files with the tool's current fingerprint. Scores, comparisons and the report are always
recomputed from the current runs, and a run that does not match the manifest is never scored.
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
from alignair_assay.core.provenance import genairr_provenance, git_state
from alignair_assay.core.reference import DEFAULT_REFERENCE, Reference
from alignair_assay.report import run_report_markdown
from alignair_assay.runners import make_runner
from alignair_assay.runners.alignair import DEFAULT_IMAGE, DEFAULT_MODEL
from alignair_assay.runners.arms import run_genotype_arm
from alignair_assay.runners.base import Runner, RunnerOptions, RunResult, absolute_path, run_suite
from alignair_assay.runners.reference import ensure_shared_reference
from alignair_assay.scoring.pipeline import load_truth, write_scores
from alignair_assay.suites import get_suite
from alignair_assay.suites.engine import DEFAULT_SEED, check_split
from alignair_assay.workflow.build import build_changes, build_suite
from alignair_assay.workflow.scoring import (
    comparison_report,
    read_genotypes,
    score_file,
    score_report,
    write_report,
)
from alignair_assay.workflow.speed import check_speed, run_speed
from alignair_assay.workflow.summary import (
    ALPHA,
    ORACLE,
    collect_run_summary,
    expected_runs,
    read_json,
    relative,
    run_status,
    run_statuses,
)

MANIFEST_KIND = "alignair_assay.manifest"
MANIFEST_VERSION = "1"


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


def runner_options(cfg: RunConfig, layout: RunLayout | None = None) -> RunnerOptions:
    """The tool options of ``cfg``. With ``layout``, the AlignAIR model cache is the run directory's
    ``runs/alignair/_work/cache``; without, it is left to the runner (the same directory once it is prepared)."""
    return RunnerOptions(igblast_dir=cfg.igblast_dir, threads=cfg.threads, alignair_image=cfg.alignair_image,
                         alignair_model=cfg.alignair_model, alignair_checkpoint=cfg.alignair_checkpoint,
                         device=cfg.device,
                         cache_dir=None if layout is None else str(layout.work_dir("alignair") / "cache"))


def make_runners(cfg: RunConfig, layout: RunLayout) -> dict[str, Runner]:
    """The oracle's and the two tools' runners, unprepared, by tool name. ValueError when a tool's options are
    missing, or a registered runner's ``name`` is not the name it is registered under (its runs would be recorded
    elsewhere)."""
    options = runner_options(cfg, layout)
    runners = {}
    for tool in dict.fromkeys((ORACLE, *cfg.tools)):
        runner = make_runner(tool, options)
        if runner.name != tool:
            raise ValueError(f"the runner registered as {tool!r} is named {runner.name!r}")
        runners[tool] = runner
    return runners


def tool_settings(runners: dict[str, Runner]) -> dict[str, dict]:
    """Per tool, its ``Runner.fingerprint()``: what it predicts with, by content (the IgBLAST binaries' digests,
    the AlignAIR image ID and model digest, the local checkpoint's digest). Each run records its tool's
    settings."""
    return {tool: runner.fingerprint() for tool, runner in runners.items()}


def identity(cfg: RunConfig, settings: dict[str, dict]) -> dict:
    """The configuration a run directory's outputs depend on: ``suites``, ``tools``, ``split``, ``reference``,
    ``seed``, ``scale`` and the tools' ``settings`` (``tool_settings``). Execution settings (``workers``,
    ``threads``, ``speed``) are left out: they change neither the cases nor the predictions."""
    return {"suites": list(cfg.suites), "tools": list(cfg.tools), "split": cfg.split, "reference": cfg.reference,
            "seed": int(cfg.seed), "scale": float(cfg.scale), "settings": settings}


def config_hash(cfg: RunConfig, layout: RunLayout) -> str:
    """Hex SHA-256 of the canonical JSON (sorted keys, no spaces) of ``identity`` of ``cfg`` with the tools'
    current fingerprints (the AlignAIR model is resolved in ``layout``'s model cache)."""
    return _digest(identity(cfg, tool_settings(make_runners(cfg, layout))))


def check_config(cfg: RunConfig) -> None:
    """ValueError for a configuration ``run_all`` cannot run, before anything is written: an unknown suite, split
    or tool, not exactly two different tools, a scale that is not positive, fewer than one worker or thread, a
    tool without the options it needs (``runners.make_runner`` on ``runner_options``: no ``--igblast-dir`` for
    igblast, no ``--alignair-checkpoint`` for alignair-local; nothing is run, pulled or downloaded), or, with
    ``speed``, what ``workflow.speed.check_speed`` refuses (S1 missing or too small to time, more threads than
    logical CPUs, a timed configuration without its tool's options)."""
    if not cfg.suites:
        raise ValueError("no suite to run")
    for name in cfg.suites:
        get_suite(name)
    if len(cfg.tools) != 2 or cfg.tools[0] == cfg.tools[1]:
        raise ValueError(f"all compares exactly two tools, a and b; got {list(cfg.tools)}")
    check_split(cfg.split)
    if not cfg.scale > 0:
        raise ValueError(f"scale must be positive, got {cfg.scale}")
    if cfg.workers < 1 or cfg.threads < 1:
        raise ValueError(f"workers and threads must be at least 1, got {cfg.workers} and {cfg.threads}")
    options = runner_options(cfg)
    for tool in cfg.tools:
        make_runner(tool, options)
    if cfg.speed:
        check_speed(cfg)


def run_all(cfg: RunConfig, layout: RunLayout, *, force: bool = False) -> int:
    """Build, run, score, compare and report ``cfg`` in the run directory ``layout``; returns 0, or 2 when any run
    failed or the oracle is not perfect on a suite (the report names which).

    1. Manifest: ``check_config``, the runners and their fingerprints (``tool_settings``; this may pull the
       AlignAIR image and install its model into the run's model cache), then write ``manifest.json``, or check the
       existing one. One made for another configuration (``config_hash``) raises ValueError naming the fields that
       differ (a tool's setting as ``<tool>.<key>``) before any output changes, unless ``force``; forced, the
       manifest is replaced (``replaced`` records the old hash and the changed fields) and the steps below redo
       what the change affects. ``warnings`` notes an AlignAIR image not pinned by digest or a model not pinned to
       a version.
    2. Build each suite (``workflow.build.build_suite``), skipped when its build is current
       (``workflow.build.build_changes`` is empty: the same identity, which covers the suite's and the truth's
       source and the GenAIRR in use). A rebuilt suite with byte-identical reads keeps the tools' runs on it, whose
       predictions depend only on the reads; the oracle's run, which records the truth file's digest, is redone
       when that file changed.
    3. Shared reference: with ``alignair`` among the tools and no shared reference yet, the image exports its
       model's reference there; then ``ensure_shared_reference`` verifies it (or writes it from the truth), so a
       model with another reference stops the run before any tool runs.
    4. Runs (``summary.expected_runs``): the oracle once per suite, then each tool on each suite and arm, through
       ``run_suite`` or ``run_genotype_arm``, recording the tool's fingerprint as the run's ``settings``. A run is
       skipped when ``run_is_current``. Every tool with a run to make is prepared before the first run.
    5. Speed, with ``cfg.speed``: ``workflow.speed.run_speed`` times every configuration of ``cfg`` on S1 and
       writes ``speed/speed.json`` with the manifest's config hash, keeping each measurement still current.
    6. Scoring, comparison and the report: ``score_and_report``.
    """
    check_config(cfg)
    reference = Reference.from_dataconfig(cfg.reference)
    layout = RunLayout(absolute_path(layout.root))
    runners = make_runners(cfg, layout)
    settings = tool_settings(runners)
    current = identity(cfg, settings)
    _manifest(cfg, current, layout, force=force)
    layout.report_json.unlink(missing_ok=True)
    layout.report_md.unlink(missing_ok=True)
    _build(cfg, layout, reference)
    _shared_reference(cfg, layout, reference, runners)
    _run(cfg, layout, reference, runners, settings)
    if cfg.speed:
        run_speed(cfg, layout, reference, config_hash=_digest(current), reuse=True)
    return score_and_report(cfg, layout, reference=reference)


def run_is_current(layout: RunLayout, tool: str, suite: str, arm: str, settings: dict) -> bool:
    """True when the run may be reused: its ``summary.run_status`` against ``settings`` is ``ok`` (return code 0,
    predictions present, the same settings, unchanged input files)."""
    return run_status(layout, tool, suite, arm, settings)["status"] == "ok"


def score_and_report(cfg: RunConfig, layout: RunLayout, *, reference: Reference | None = None,
                     command: str = "all") -> int:
    """Score, compare and report the runs of ``cfg`` in ``layout``; returns 0, or 2 when a run failed, is
    missing, stale or unscorable, the oracle is not perfect on a suite, or a speed measurement of the
    configuration failed.

    For every suite and arm the old ``scores/<S>/<arm>/`` is removed. Each run whose ``summary.run_statuses``
    entry is ``ok`` (``require_successful_run`` passes, the run was made with the manifest's settings for its tool,
    on unchanged files of a suite whose build matches ``cfg``) is scored as the ``score`` command scores it:
    ``<tool>.scores.jsonl.gz``, ``<tool>.report.json`` and ``<tool>.report.md``, with the suite's primary metrics
    leading the markdown. A suite with a ``donors.json`` is scored as ``score --donors`` scores it: against each
    read's donor genotype, with the genotype metrics and the excess-alleles table. The oracle is scored on the
    shared arm. When both tools are scored, ``comparison.json`` and ``comparison.md`` are written as the
    ``compare`` command writes them (Holm at ``summary.ALPHA``). Then ``report.json``
    (``summary.collect_run_summary``) and ``report.md`` (``report.run.run_report_markdown``) are written.
    ``command`` is recorded in the reports' provenance.
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
    for warning in summary["truth_warnings"]:
        counts = warning["unscorable"]
        print(f"warning: {warning['suite']}: the truth of {sum(counts.values())} segments "
              f"({' '.join(f'{g}={n}' for g, n in counts.items())}) could not be built, so they are not scored; see "
              f"{relative(layout, layout.build_json(warning['suite']))}", file=sys.stderr)
    speed = summary["speed"] or {}
    for failure in speed.get("failed") or []:
        print(f"error: speed {failure['name']} failed: {failure['error']}", file=sys.stderr)
    if speed.get("stale"):
        print(f"warning: {relative(layout, layout.speed_json)} was {speed['reason']} and is not reported; run all "
              "--speed to measure this configuration", file=sys.stderr)
    for row in speed.get("stale_results") or []:
        print(f"warning: speed {row['name']} was {row['reason']} and is not reported; run all --speed to measure "
              "it again", file=sys.stderr)
    return 2 if summary["failed_runs"] or summary["oracle_failures"] or speed.get("failed") else 0


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


def _flat(fields: dict) -> dict:
    """An identity with each tool's setting as its own ``<tool>.<key>`` field, for naming what differs."""
    flat = {name: value for name, value in fields.items() if name != "settings"}
    for tool, values in (fields.get("settings") or {}).items():
        flat.update({f"{tool}.{key}": value for key, value in values.items()})
    return flat


def _manifest(cfg: RunConfig, current: dict, layout: RunLayout, *, force: bool) -> None:
    """Write the manifest, keep an identical one, or refuse (ValueError) one made for another configuration."""
    digest = _digest(current)
    replaced = None
    if layout.manifest.is_file():
        old = read_json(layout.manifest) or {}
        if old.get("config_hash") == digest:
            return
        previous, now = _flat(old.get("identity") or {}), _flat(current)
        changed = [name for name in sorted(previous.keys() | now.keys()) if previous.get(name) != now.get(name)]
        if not force:
            differences = "; ".join(f"{name} {previous.get(name)} -> {now.get(name)}" for name in changed)
            raise ValueError(f"{layout.manifest} was written for another configuration ({differences}). Rerun with "
                             "that configuration, choose another --run-dir, or pass --force to rebuild and re-run "
                             "what the change affects")
        replaced = {"config_hash": old.get("config_hash"), "changed": changed}
    warnings = _pinning_warnings(cfg)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    manifest = {
        "kind": MANIFEST_KIND, "version": MANIFEST_VERSION, "config": dataclasses.asdict(cfg), "identity": current,
        "config_hash": digest, "split": cfg.split,
        "host": {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
                 "logical_cpus": os.cpu_count()},
        "git": git_state(Path(__file__).resolve().parent), "genairr": genairr_provenance(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "replaced": replaced,
        "warnings": warnings,
    }
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _pinning_warnings(cfg: RunConfig) -> list[str]:
    """Notes for an AlignAIR image not pinned by digest and a registry model not pinned to a version: their
    resolved image ID and model digest are in the configuration, but the references alone do not say what ran."""
    if "alignair" not in cfg.tools:
        return []
    warnings = []
    image = cfg.alignair_image or DEFAULT_IMAGE
    if "@sha256:" not in image:
        warnings.append(f"the AlignAIR image {image} is not pinned by digest; the image ID it resolved to is "
                        "recorded as alignair.image_id")
    if "@" not in cfg.alignair_model:
        warnings.append(f"the AlignAIR model {cfg.alignair_model} is not pinned to a version; the model file's "
                        "SHA-256 is recorded as alignair.model_sha256")
    return warnings


def _build(cfg: RunConfig, layout: RunLayout, reference: Reference) -> None:
    """Build every suite whose build is not current, saying what changed (``workflow.build.build_changes``)."""
    digest = reference.digest()
    for name in cfg.suites:
        changes = build_changes(layout, name, split=cfg.split, reference_digest=digest, seed=cfg.seed,
                                scale=cfg.scale)
        if not changes:
            print(f"build {name}: up to date, skipped", flush=True)
            continue
        build = build_suite(name, layout, reference_name=cfg.reference, split=cfg.split, seed=cfg.seed,
                            scale=cfg.scale, workers=cfg.workers)
        why = "" if changes == ["no build record"] else f" (rebuilt; changed: {', '.join(changes)})"
        print(f"build {name}: {build['n_cases']} cases in {build['seconds']:.1f} s{why}", flush=True)


def _shared_reference(cfg: RunConfig, layout: RunLayout, reference: Reference, runners: dict[str, Runner]) -> None:
    """Export AlignAIR's reference as the shared one when it is a tool and there is none yet, then verify the
    shared reference (or write it from the truth). An exported reference that fails verification is removed, so a
    later run with another model exports again."""
    exported = False
    if "alignair" in cfg.tools and not layout.reference_fasta.is_file():
        layout.reference_dir.mkdir(parents=True, exist_ok=True)
        runners["alignair"].export_reference(layout.reference_fasta)
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


def _run(cfg: RunConfig, layout: RunLayout, reference: Reference, runners: dict[str, Runner],
         settings: dict[str, dict]) -> None:
    """Make every expected run that is not current; every tool with a run to make is prepared first."""
    pending = []
    for tool, suite, arm in expected_runs(cfg):
        if run_is_current(layout, tool, suite, arm, settings[tool]):
            print(f"run {tool} {suite}/{arm}: up to date, skipped", flush=True)
        else:
            pending.append((tool, suite, arm))
    for tool in dict.fromkeys(tool for tool, _, _ in pending):
        runners[tool].prepare(layout, reference)
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
    """Write every suite and arm's scores and comparison; returns the runs that are ``ok`` but could not be scored
    (``run_status``-shaped, status ``not scored``), every run of a suite whose ``donors.json`` cannot be read
    among them."""
    a, b = cfg.tools
    statuses = run_statuses(layout, cfg)
    errors = []
    for name in cfg.suites:
        suite = get_suite(name)
        header, truths = load_truth(layout.truth(name), cfg.reference, reference)
        # A suite with donors is scored against each read's donor genotype, recorded as an input.
        donors = layout.donors(name) if layout.donors(name).is_file() else None
        genotypes, donors_error = None, None
        if donors is not None:
            try:
                genotypes = read_genotypes(donors)
            except (OSError, ValueError) as error:
                donors_error = f"{relative(layout, donors)} cannot be read, so the suite is not scored: {error}"
        extra_inputs = {} if donors is None else {"donors": donors}
        for arm in suite.arms:
            shutil.rmtree(layout.scores_dir(name, arm), ignore_errors=True)
            layout.scores_dir(name, arm).mkdir(parents=True)
            scored = {}
            for tool in dict.fromkeys(([ORACLE] if arm == "shared" else []) + [a, b]):
                status = statuses[tool, name, arm]
                if status["status"] != "ok":
                    continue  # the summary lists it as a failed run
                if donors_error is not None:
                    errors.append({**status, "status": "not scored", "error": donors_error})
                    continue
                predictions = layout.predictions(tool, name, arm)
                try:
                    scored[tool] = score_file(tool, truths, predictions, reference, genotypes=genotypes)
                except (FileNotFoundError, ValueError) as error:
                    errors.append({**status, "status": "not scored", "error": str(error)})
                    continue
                report = score_report(scored[tool], truths, header, command=command,
                                      inputs={"truth": layout.truth(name), "predictions": predictions,
                                              "run": layout.run_json(tool, name, arm), **extra_inputs},
                                      reference_name=cfg.reference)
                write_scores(scored[tool].scores, layout.tool_scores(name, arm, tool))
                write_report(report, layout.tool_report_json(name, arm, tool), layout.tool_report_md(name, arm, tool),
                             headline=suite.primary)
            compared = a in scored and b in scored
            if compared:
                report = comparison_report(scored[a], scored[b], truths, header, command=command,
                                           inputs={"truth": layout.truth(name), "a": layout.predictions(a, name, arm),
                                                   "b": layout.predictions(b, name, arm), **extra_inputs},
                                           reference_name=cfg.reference, alpha=ALPHA)
                write_report(report, layout.comparison_json(name, arm), layout.comparison_md(name, arm),
                             headline=suite.primary)
            print(f"score {name}/{arm}: scored {', '.join(scored) or 'nothing'}; "
                  + (f"compared {a} with {b} in {relative(layout, layout.comparison_md(name, arm))}" if compared
                     else "not compared"), flush=True)
    return errors
