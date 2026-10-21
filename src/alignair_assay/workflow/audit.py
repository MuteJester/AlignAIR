"""``audit``: the readiness gate for model comparisons.

``run_audit(cfg, layout)`` proves, before any AlignAIR model work, that the assay can be trusted to compare two
checkpoints: the truth checks itself, its outputs reproduce, a model compared with itself differs by exactly nothing
on every surface (``all``, ``diagnose``, ``real``), and IgBLAST lands where independent evidence says it should. It
writes ``layout.audit_json`` and ``layout.audit_md`` (``report.audit``), whose first line is the verdict.

**Checks.** Each is ``{name, status, n, evidence, reason}``: ``status`` is ``pass``, ``fail``, ``warn`` (worth a
look, never a failure) or ``skipped`` (not checkable now, with why); ``n`` is how many reads or cases it rests on;
``evidence`` holds the numbers, its ``summary`` a one-line digest of them. A check that passes on nothing (``n`` 0)
fails (``settle``), and so does a check that raises (its error is the reason). The ``INFORMATIONAL`` checks warn at
worst; every other one gates (``GATING``: it can fail the audit). The audit passes when no check fails, no gating
check was skipped and at least one passed (``verdict``); a skipped gating check makes it ``incomplete`` (exit 3),
since what was not checked may have failed, and an audit that checked nothing never passes.

- Self-checks:
  - ``self_check.sim_oracle``: the stored ``all`` run's oracle is perfect on every suite (``workflow.summary
    .oracle_check`` of its stored oracle scores, read only);
  - ``self_check.sim_oracle_current``: the oracle of the null ``all`` run below, on a fresh dev build at this code;
  - ``self_check.sim_staleness``: the stored run's suite builds against this checkout's suite, truth and GenAIRR
    code (``workflow.build.build_changes``); informational, ``warn`` when they moved;
  - ``self_check.real_oracle``: every dev donor of the stored ``real`` run passed its oracle self-check (the pipeline
    path and the decoys included);
  - ``self_check.real_truth_health``: the truth-health flags of those donors, ``warn`` when there are any.
- Reproducibility (in scratch):
  - ``reproducible.sim_build``: ``REPRO_SUITE`` built twice at ``REPRO_SCALE`` on the dev split; its cases, truth and
    reads must be byte-identical. Compared only when GenAIRR's source did not move during the two builds
    (``genairr_digest`` before, between and after); otherwise ``skipped (GenAIRR moved)``, never a failure;
  - ``reproducible.real_scores``: one dev donor's truth and every run's scores recomputed from the stored reads,
    genotype and predictions, and its self-check redone; each file's content (its rows, and its header without the
    code digest and the fingerprint that holds it) must equal the stored one's (``compare_records``). Its bytes are
    reported, not required: files written before JSON-lines gzip was reproducible carry a time in their gzip header,
    and the scores' inputs name the truth file by those bytes, so a stored truth's digest stands for the recomputed
    one's when the two hold the same content.
- Null comparisons (one checkpoint under ``NULL_LABELS``): every Δ 0, no significant verdict, no read won or lost
  (fixed = broken = 0), each side's own scores equal and every read either side scores paired (``compare_problems``);
  the comparison must pair something (``pairing``): a comparison that pairs nothing is not null, it is empty; and it
  must hold everything expected of it, since what is missing is never null: every end-to-end headline metric
  (``SIM_HEADLINE``, ``REAL_HEADLINE``) on every suite and arm, and in ``diagnose`` every suite diagnosed, each with
  its head family (one metric per ``HEAD_BOUNDARY_PREFIXES`` at least) and its scored reads. Its ``n`` is the fewest
  reads a headline metric pairs:
  - ``null.diagnose``: the checkpoint file against itself loaded in memory (two model keys, each diagnosed) on fresh
    dev builds of ``DIAGNOSE_SUITES``;
  - ``null.all``: ``all --tools alignair-local@a,alignair-local@b`` on fresh dev builds of ``NULL_SUITES`` at
    ``NULL_SCALE``;
  - ``null.real``: ``real --tools alignair-api@a,alignair-api@b --cap REAL_CAP`` on ``REAL_DONOR``.
- IgBLAST bands:
  - ``band.real_known``: IgBLAST's ``real.v.known_top1`` and ``real.v.ref_known_top1`` on the shared arm at least
    ``KNOWN_FLOOR`` on every dev donor of the stored ``real`` run;
  - ``band.sim_v_top1_current``: its S1 ``v.top1_in_set`` at this code, IgBLAST run now on a fresh dev S1, within
    ``V_TOP1_CURRENT_BAND`` of ``V_TOP1_TARGET``; ``fail`` outside it, ``skipped`` without an IgBLAST release (which
    leaves the audit incomplete);
  - ``band.sim_v_top1`` (informational): the same metric in the stored ``all`` run, consistent with the P2 record
    when within ``V_TOP1_BAND`` of ``V_TOP1_TARGET`` (ruling A1: read as recorded), ``warn`` otherwise: it was
    computed by older code, and the P2 number is where the target comes from;
  - ``band.speed`` (informational): IgBLAST's steady throughput, timed now on a fresh dev S1 by the speed protocol,
    within ``SPEED_BAND`` of the stored run's measurement; ``skipped (load)`` when the 1-minute load stays above
    ``SPEED_LOAD``, ``warn`` outside the band or on an error, never a failure.

**Read only.** The stored runs are only read: the test-split ``all`` run for its computed oracle and IgBLAST numbers
(it is never run or scored again), the ``real`` run for its records. Anything re-run goes into a fresh
``layout.audit_scratch(<check>)``, removed first, and the output directory may not overlap either run.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.freeze import frozen_record
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import CODE_SUFFIXES, genairr_provenance, git_state, sha256_file, tree_digest
from alignair_assay.core.records import read_json, write_json, write_text
from alignair_assay.core.reference import DEFAULT_REFERENCE, Reference
from alignair_assay.real.fetch import PILOT_SPLIT
from alignair_assay.real.genotype import genotype_from_json, merge_identical
from alignair_assay.report.audit import audit_markdown
from alignair_assay.runners import make_runner
from alignair_assay.runners.base import RunnerOptions, absolute_path, read_run_json, run_suite
from alignair_assay.runners.reference import ensure_shared_reference
from alignair_assay.scoring.pipeline import load_truth, truth_reference
from alignair_assay.suites import get_suite
from alignair_assay.suites.donors import read_genotypes
from alignair_assay.suites.engine import DEFAULT_SEED
from alignair_assay.truth.io import read_truth_header
from alignair_assay.workflow import real_score, real_steps
from alignair_assay.workflow.build import GENAIRR_IDENTITY, build_changes, build_suite
from alignair_assay.workflow.orchestrate import RunConfig, read_run_config, run_all
from alignair_assay.workflow.real import RealConfig, donor_scoring, run_real, scoring_inputs, self_check
from alignair_assay.workflow.scoring import score_file
from alignair_assay.workflow.speed import run_speed
from alignair_assay.workflow.summary import ORACLE, oracle_check

AUDIT_KIND = "alignair_assay.audit"
AUDIT_VERSION = "2"  # 2: the provenance names the code audit_code_sha256, taken at the start
STATUSES = ("pass", "fail", "warn", "skipped")
# The check groups, in report order, and every check in report order.
GROUPS = {"self_check": "Self-checks", "reproducible": "Reproducibility", "null": "Null comparisons",
          "band": "IgBLAST bands"}
CHECKS = ("self_check.sim_oracle", "self_check.sim_oracle_current", "self_check.sim_staleness",
          "self_check.real_oracle", "self_check.real_truth_health", "reproducible.sim_build",
          "reproducible.real_scores", "null.diagnose", "null.all", "null.real", "band.real_known",
          "band.sim_v_top1", "band.sim_v_top1_current", "band.speed")
# The informational checks: a warning at worst, never a failure, even when they raise.
INFORMATIONAL = ("self_check.sim_staleness", "band.sim_v_top1", "band.speed")
# The checks that can fail the audit (every other one), in report order: one of them skipped leaves it incomplete.
GATING = tuple(name for name in CHECKS if name not in INFORMATIONAL)
# The verdicts, and the command's exit status for each.
VERDICTS = {"pass": 0, "fail": 2, "incomplete": 3}
# The audited checkpoint's label (``--checkpoint base=PATH``) and the two labels it runs under in a null comparison.
CHECKPOINT_LABEL = "base"
NULL_LABELS = ("a", "b")
# Reproducibility: the suite built twice, its scale, and the files both builds must hold (donors.json only with donors).
REPRO_SUITE, REPRO_SCALE = "S1", 0.02
REPRO_FILES = ("cases", "truth", "reads")
# The null ``all`` comparison's suites and scale; ``diagnose``'s; ``real``'s donor and cap.
NULL_SUITES, NULL_SCALE = ("S1",), 0.05
DIAGNOSE_SUITES, DIAGNOSE_SCALE = ("S1", "S3"), 0.05
REAL_DONOR, REAL_CAP = "W-97", 2000
# IgBLAST against independent evidence: its known answers on real reads, its S1 V top-1 (the P2 number) and speed.
KNOWN_METRICS = ("real.v.known_top1", "real.v.ref_known_top1")
KNOWN_FLOOR = 0.99
V_TOP1_SUITE, V_TOP1_METRIC = "S1", "v.top1_in_set"
V_TOP1_TARGET, V_TOP1_BAND = 0.979, 0.01
# IgBLAST's S1 V top-1 at this code (the gate): its band around the same target, and the scale of the dev S1 it is
# run on when the speed check built none.
V_TOP1_CURRENT_BAND, V_TOP1_CURRENT_SCALE = 0.015, 0.2
IGBLAST = "igblast"
SPEED_BAND = 0.15
# The 1-minute load above which the machine is too busy to time; how long to wait for it to settle, polled how often.
SPEED_LOAD = 1.5
SPEED_SETTLE_SECONDS = 180
SPEED_POLL_SECONDS = 5
# The scale of the fresh dev S1 the speed protocol times (as the stored run's: N_FULL of its reads, evenly spaced).
SPEED_SCALE = 1.0
SPEED_SUITE = "S1"
# A JSON-lines header's fields that name the code it was computed by: what may differ when only the code moved.
CODE_FIELDS = ("code_sha256", "fingerprint")
_EXAMPLES = 8
_PACKAGE = Path(__file__).resolve().parents[1]
_TIE = "tie_or_inconclusive"
# The end-to-end headline metrics a null comparison's n rests on: the fewest reads any of them pairs.
SIM_HEADLINE = ("clone_key.exact", "junction.nt_exact", "v.top1_in_set", "d.top1_in_set", "j.top1_in_set")
REAL_HEADLINE = ("real.v.unsupported", "real.v.known_top1", "real.v.ref_known_top1")
# The head family of a diagnose comparison holds a boundary metric of each gene and edge; one pairs only the reads
# both models orient as the truth, so it must pair some, not all.
HEAD_BOUNDARY_PREFIXES = tuple(f"head.{gene}.{edge}_" for gene in "vdj" for edge in ("start", "end"))


@dataclass(frozen=True)
class AuditConfig:
    """One audit: the stored ``all`` run (``sim_run``) and ``real`` run, the data root of the real reads, the audited
    ``checkpoint`` (a ``.alignair`` file) and the torch ``device`` the null comparisons run it on, whether to time
    IgBLAST (``skip_speed``), the IgBLAST release to time (None: the one the ``all`` run recorded), the build
    ``workers`` and tool ``threads``, the suites' ``seed`` and ``reference``; then the ruling's sizes, which only a
    test shrinks: the reproducibility build, the null comparisons' suites, scales, donor and cap, the diagnosed
    ``diagnose_variants`` (None: every one), the speed protocol's S1 scale and the scale of the S1 IgBLAST's current
    V top-1 is measured on when the speed check built none."""

    sim_run: str
    real_run: str
    data_root: str
    checkpoint: str
    device: str = "cpu"
    skip_speed: bool = False
    igblast_dir: str | None = None
    workers: int = 8
    threads: int = 8
    seed: int = DEFAULT_SEED
    reference: str = DEFAULT_REFERENCE
    repro_suite: str = REPRO_SUITE
    repro_scale: float = REPRO_SCALE
    null_suites: tuple[str, ...] = NULL_SUITES
    null_scale: float = NULL_SCALE
    diagnose_suites: tuple[str, ...] = DIAGNOSE_SUITES
    diagnose_scale: float = DIAGNOSE_SCALE
    diagnose_variants: tuple[str, ...] | None = None
    real_donor: str = REAL_DONOR
    real_cap: int = REAL_CAP
    speed_scale: float = SPEED_SCALE
    v_top1_scale: float = V_TOP1_CURRENT_SCALE


# --- checks -------------------------------------------------------------------------------------------------------

def check(name: str, status: str, n: int, evidence: dict, reason: str | None = None) -> dict:
    """A check record: ``name`` (of ``CHECKS``), ``status`` (of ``STATUSES``), ``n`` (the reads or cases it rests
    on), ``evidence`` (with a one-line ``summary``) and the ``reason`` for a status other than pass."""
    if status not in STATUSES:
        raise ValueError(f"unknown check status {status!r}")
    return {"name": name, "status": status, "n": int(n), "evidence": evidence, "reason": reason}


def settle(found: dict) -> dict:
    """``found``, failed when it passed on nothing (``n`` below 1): a check that checked nothing never passes."""
    if found["status"] == "pass" and found["n"] < 1:
        return {**found, "status": "fail", "reason": "checked nothing (n = 0)"}
    return found


def verdict(checks: list[dict]) -> dict:
    """The audit's verdict (of ``VERDICTS``): ``fail`` when a check failed; else ``incomplete`` when a check that can
    fail the audit (``GATING``) was skipped, since what was not checked may have failed; else ``fail`` when no check
    passed (an audit that checked nothing never passes); else ``pass``. With the ``counts`` per status, the
    ``failed`` and ``skipped_gating`` check names and the one-line ``line`` that opens ``audit.md``."""
    counts = Counter(found["status"] for found in checks)
    failed = [found["name"] for found in checks if found["status"] == "fail"]
    skipped = [found["name"] for found in checks if found["status"] == "skipped" and found["name"] in GATING]
    checked = any(found["status"] == "pass" for found in checks)
    status = "fail" if failed else "incomplete" if skipped else "pass" if checked else "fail"
    line = (f"AUDIT {status.upper()}: {len(checks)} checks, " +
            ", ".join(f"{counts[name]} {name}" for name in STATUSES))
    if failed:
        line += f"; failed: {', '.join(failed)}"
    if skipped:
        line += f"; skipped, and able to fail the audit: {', '.join(skipped)}"
    if not checked:
        line += "; nothing was checked"
    return {"status": status, "counts": {name: counts[name] for name in STATUSES}, "failed": failed,
            "skipped_gating": skipped, "line": line}


# --- self-checks of the stored runs ---------------------------------------------------------------------------------

def sim_oracle(sim: RunLayout) -> dict:
    """``self_check.sim_oracle``: the oracle rule on the stored oracle scores of every suite of the ``all`` run
    (``workflow.summary.oracle_check``, read only), beside what its ``report.json`` recorded."""
    cfg = read_run_config(sim)
    recorded = {suite["name"]: suite.get("oracle") or {} for suite in (read_json(sim.report_json) or {}).get(
        "suites") or []}
    suites, bad, disagree = {}, [], []
    for name in cfg.suites:
        found = oracle_check(sim, name)
        suites[name] = {"status": found["status"], "cases": found["cases"], "problems": found["problems"],
                        "recorded": (recorded.get(name) or {}).get("status")}
        if found["status"] != "perfect":
            bad.append(name)
        if suites[name]["recorded"] != found["status"]:
            disagree.append(f"{name} (recorded {suites[name]['recorded']}, scores {found['status']})")
    cases = sum(entry["cases"] for entry in suites.values())
    summary = (f"oracle perfect on {len(suites) - len(bad)} of {len(suites)} suites ({cases} cases, "
               f"{cfg.split} split)" + (f"; the report disagrees on {len(disagree)}" if disagree else
                                         ", as its report recorded"))
    evidence = {"summary": summary, "run": str(sim.root), "split": cfg.split, "suites": suites}
    reasons = ([f"the oracle is not perfect on {', '.join(bad)}: its truth has a defect or is unchecked"] if bad
               else []) + ([f"the run's report recorded another oracle status than its scores give: "
                            f"{', '.join(disagree)}"] if disagree else [])
    if reasons:
        return check("self_check.sim_oracle", "fail", cases, evidence, "; ".join(reasons))
    return check("self_check.sim_oracle", "pass", cases, evidence)


def sim_staleness(sim: RunLayout) -> dict:
    """``self_check.sim_staleness`` (informational): the stored run's suite builds against this checkout's suite,
    truth and GenAIRR code (``workflow.build.build_changes``, which reads only), and its recorded assay commit and
    GenAIRR against the current ones; ``warn`` when a build is not current."""
    cfg = read_run_config(sim)
    manifest = read_json(sim.manifest) or {}
    digest = Reference.from_dataconfig(cfg.reference).digest()
    changes = {name: build_changes(sim, name, split=cfg.split, reference_digest=digest, seed=cfg.seed,
                                   scale=cfg.scale) for name in cfg.suites}
    stale = {name: found for name, found in changes.items() if found}
    current_genairr = genairr_provenance()
    evidence = {"run": str(sim.root), "split": cfg.split, "stale_builds": stale,
                "assay": {"recorded": manifest.get("git"), "current": git_state(_PACKAGE)},
                "genairr": {"recorded": {key: (manifest.get("genairr") or {}).get(key) for key in GENAIRR_IDENTITY},
                            "current": {key: current_genairr.get(key) for key in GENAIRR_IDENTITY}}}
    if stale:
        fields = sorted({field for found in stale.values() for field in found})
        evidence["summary"] = (f"{len(stale)} of {len(changes)} suite builds are not current for this checkout "
                               f"(changed: {', '.join(fields)})")
        return check("self_check.sim_staleness", "warn", len(changes), evidence,
                     "informational: the stored run's numbers were computed by other suite, truth or GenAIRR code "
                     "than this checkout's; the audit reads them as recorded and never re-scores the test split")
    evidence["summary"] = f"every one of the {len(changes)} suite builds is current for this checkout"
    return check("self_check.sim_staleness", "pass", len(changes), evidence)


def dev_donors(summary: dict) -> tuple[dict[str, dict], list[str]]:
    """The blocks of the pilot's dev donors (``real.fetch.PILOT_SPLIT``) in a ``real`` summary, in the pilot's order,
    and the dev donors the summary lacks: every donor check covers every dev donor, not only those a run holds."""
    blocks = summary.get("donors") or {}
    return ({donor: blocks[donor] for donor in PILOT_SPLIT["dev"] if donor in blocks},
            [donor for donor in PILOT_SPLIT["dev"] if donor not in blocks])


def _missing(donors: list[str]) -> list[str]:
    """The reason a donor check fails for dev donors the stored run lacks; none when it lacks none."""
    return [f"the dev donors {', '.join(donors)} are missing from the stored run"] if donors else []


def real_summary(real: RunLayout) -> dict:
    """The stored ``real`` run's summary; FileNotFoundError without one."""
    summary = read_json(real.real_summary_json())
    if summary is None:
        raise FileNotFoundError(f"no readable {real.real_summary_json()}: the run directory was not made by `real`")
    return summary


def real_oracle(real: RunLayout) -> dict:
    """``self_check.real_oracle``: every dev donor of the stored ``real`` run scored and its oracle self-check
    ``passed``, the pipeline path and the decoys included; a dev donor missing from the run fails."""
    donors, missing = dev_donors(real_summary(real))
    rows, bad = {}, []
    for donor, block in donors.items():
        found = block.get("oracle_self_check") or {}
        pipeline, decoys = found.get("pipeline") or {}, found.get("decoys") or {}
        rows[donor] = {"status": found.get("status") if block.get("status") == "done" else "stopped",
                       "scored": found.get("scored") or 0, "reads": found.get("reads"),
                       "ref_reads": found.get("ref_reads"), "pipeline_agrees": pipeline.get("agrees"),
                       "pipeline_reads": pipeline.get("reads"), "decoys_passed": decoys.get("passed")}
        if rows[donor]["status"] != "passed" or pipeline.get("passed") is not True or decoys.get("passed") is not True:
            bad.append(donor)
    scored = sum(row["scored"] for row in rows.values())
    evidence = {"summary": f"self-check passed on {len(rows) - len(bad)} of {len(PILOT_SPLIT['dev'])} dev donors "
                           f"({scored} oracle reads scored, pipeline path and decoys included)",
                "run": str(real.root), "donors": rows, "missing": missing}
    reasons = _missing(missing) + ([f"the truth self-check did not pass on {', '.join(bad)}"] if bad else [])
    if reasons:
        return check("self_check.real_oracle", "fail", scored, evidence, "; ".join(reasons))
    return check("self_check.real_oracle", "pass", scored, evidence)


def real_truth_health(real: RunLayout) -> dict:
    """``self_check.real_truth_health``: each dev donor's truth-health flags (carried genes whose reference-exact
    reads no donor allele explains, ``real.truth.truth_health``); ``warn`` when any is flagged or a donor has no
    truth-health block (never audited, so not clean); a dev donor missing from the run fails."""
    donors, missing = dev_donors(real_summary(real))
    flags, reads, unaudited = {}, 0, []
    for donor, block in donors.items():
        if not isinstance(block.get("truth_health"), dict):
            unaudited.append(donor)
            continue
        health = block["truth_health"]
        reads += health.get("reads") or 0
        genes = {entry["gene"]: entry for entry in health.get("genes") or []}
        for gene in health.get("flagged") or []:
            entry = genes.get(gene) or {}
            flags.setdefault(donor, []).append({"gene": gene, "reads": entry.get("reads"),
                                                "clonotypes": entry.get("clonotypes"),
                                                "alleles": entry.get("alleles"), "h0_only": entry.get("h0_only")})
    listed = "; ".join(f"{donor} {', '.join(flag['gene'] for flag in found)}" for donor, found in flags.items())
    evidence = {"summary": (f"{sum(map(len, flags.values()))} flags on {len(flags)} of {len(PILOT_SPLIT['dev'])} dev "
                            "donors" + (f": {listed}" if listed else "")
                            + (f"; no truth-health block for {', '.join(unaudited)}" if unaudited else "")),
                "flags": flags, "donors": list(donors), "unaudited": unaudited, "missing": missing}
    if missing:
        return check("self_check.real_truth_health", "fail", reads, evidence, "; ".join(_missing(missing)))
    warnings = ([f"{', '.join(unaudited)} have no truth-health block: their truth was never audited"] if unaudited
                else []) + (["carried genes whose reference-exact reads no donor allele explains: look at them before "
                             "trusting those genes' calls"] if flags else [])
    if warnings:
        return check("self_check.real_truth_health", "warn", reads, evidence,
                     "; ".join(warnings) + " (a warning, never a failure)")
    return check("self_check.real_truth_health", "pass", reads, evidence)


# --- IgBLAST bands ------------------------------------------------------------------------------------------------

def real_known_band(real: RunLayout) -> dict:
    """``band.real_known``: IgBLAST's ``KNOWN_METRICS`` on the shared arm at least ``KNOWN_FLOOR`` on every dev donor
    of the stored ``real`` run (its documented and reference-exact known answers); a dev donor missing from the run
    fails."""
    donors, missing = dev_donors(real_summary(real))
    rows, bad, reads = {}, [], 0
    for donor, block in donors.items():
        summary = ((((block.get("tools") or {}).get("shared") or {}).get(IGBLAST) or {}).get("summary") or {})
        rows[donor] = {metric: {"mean": (summary.get(metric) or {}).get("mean"), "n": (summary.get(metric) or {})
                                .get("n", 0)} for metric in KNOWN_METRICS}
        reads += rows[donor][KNOWN_METRICS[0]]["n"] or 0
        low = [metric for metric, entry in rows[donor].items() if entry["mean"] is None or entry["mean"] < KNOWN_FLOOR]
        if low:
            bad.append(f"{donor} ({', '.join(low)})")
    evidence = {"summary": "; ".join(f"{donor} " + ", ".join(f"{metric.removeprefix('real.v.')} "
                                                             f"{_percent(entry['mean'])} (n={entry['n']})"
                                                             for metric, entry in found.items())
                                     for donor, found in rows.items()) or "no dev donor",
                "floor": KNOWN_FLOOR, "arm": "shared", "donors": rows, "missing": missing}
    reasons = _missing(missing) + ([f"IgBLAST is below {_percent(KNOWN_FLOOR)} on {'; '.join(bad)}"] if bad else [])
    if reasons:
        return check("band.real_known", "fail", reads, evidence, "; ".join(reasons))
    return check("band.real_known", "pass", reads, evidence)


def sim_v_top1_band(sim: RunLayout) -> dict:
    """``band.sim_v_top1`` (informational): IgBLAST's S1 ``v.top1_in_set`` in the stored ``all`` run (its score
    report, read only), consistent with the P2 record when within ``V_TOP1_BAND`` of ``V_TOP1_TARGET``; ``warn``
    outside it or without IgBLAST's S1 scores. The stored number was computed by older code (and is where the target
    comes from), so it never fails the audit: ``band.sim_v_top1_current`` is the gate."""
    name = "band.sim_v_top1"
    cfg = read_run_config(sim)
    if IGBLAST not in cfg.tools or V_TOP1_SUITE not in cfg.suites:
        return check(name, "warn", 0, {"summary": "IgBLAST on S1 is not in the run", "tools": list(cfg.tools),
                                       "suites": list(cfg.suites)},
                     f"informational: the stored run has no {IGBLAST} scores on {V_TOP1_SUITE}")
    report = read_json(sim.tool_report_json(V_TOP1_SUITE, "shared", IGBLAST)) or {}
    entry = (report.get("overall") or {}).get(V_TOP1_METRIC) or {}
    mean, n = entry.get("mean"), entry.get("n") or 0
    consistent = mean is not None and abs(mean - V_TOP1_TARGET) <= V_TOP1_BAND + 1e-12
    evidence = {"summary": f"IgBLAST S1 {V_TOP1_METRIC} {_percent(mean, 2)} (n={n}) in the stored run, "
                           f"{'' if consistent else 'not '}consistent with the P2 record ({_percent(V_TOP1_TARGET)} ± "
                           f"{100 * V_TOP1_BAND:.0f} pp)", "mean": mean, "target": V_TOP1_TARGET,
                "band": V_TOP1_BAND, "split": cfg.split, "run": str(sim.root)}
    if not consistent:
        return check(name, "warn", n, evidence,
                     f"informational: IgBLAST's S1 {V_TOP1_METRIC} {_percent(mean, 2)} in the stored run is outside "
                     f"{_percent(V_TOP1_TARGET)} ± {100 * V_TOP1_BAND:.0f} pp of the P2 record")
    return check(name, "pass", n, evidence)


def igblast_speed(cfg: AuditConfig, sim: RunLayout, scratch: RunLayout, *, load: Callable[[], float] | None = None,
                  sleep: Callable[[float], None] = time.sleep) -> dict:
    """``band.speed`` (informational): IgBLAST timed now by the speed protocol (``workflow.speed.run_speed``, its
    configurations the stored run measured, e.g. ``igblast_1t`` and ``igblast_8t``) on a fresh dev S1 at
    ``speed_scale`` in ``scratch``, against the stored run's steady throughput: ``pass`` within ``SPEED_BAND``,
    ``warn`` outside it or when a measurement fails. ``skipped`` with ``--skip-speed``, without a stored measurement
    or IgBLAST release, and ``skipped (load)`` when the 1-minute ``load`` (``os.getloadavg``) stays above
    ``SPEED_LOAD`` for ``SPEED_SETTLE_SECONDS`` before the build and again before the timing. Anything that raises
    while it builds or times is a ``warn`` too: speed never fails the audit."""
    try:
        return _igblast_speed(cfg, sim, scratch, load=load or (lambda: os.getloadavg()[0]), sleep=sleep)
    except Exception as error:  # informational: an error timing IgBLAST is worth a look, never a failure
        return check("band.speed", "warn", 0, {"summary": f"not timed: {type(error).__name__}: {error}"},
                     f"informational: timing IgBLAST raised {type(error).__name__}: {error}")


def _igblast_speed(cfg: AuditConfig, sim: RunLayout, scratch: RunLayout, *, load: Callable[[], float],
                   sleep: Callable[[float], None]) -> dict:
    name = "band.speed"
    if cfg.skip_speed:
        return check(name, "skipped", 0, {"summary": "not timed (--skip-speed)"}, "skipped: --skip-speed")
    speed = (read_json(sim.report_json) or {}).get("speed") or {}
    stored = {config: result for config, result in (speed.get("results") or {}).items()
              if result.get("tool") == IGBLAST and "error" not in result and result.get("steady_reads_per_s")}
    if speed.get("stale") or not stored:
        return check(name, "skipped", 0, {"summary": "no current IgBLAST speed measurement in the stored run"},
                     "skipped: the stored run has no current IgBLAST speed measurement to compare with")
    igblast_dir = cfg.igblast_dir or read_run_config(sim).igblast_dir
    if not igblast_dir or not Path(igblast_dir).is_dir():
        return check(name, "skipped", 0, {"summary": f"no IgBLAST release ({igblast_dir})"},
                     f"skipped: no IgBLAST release directory {igblast_dir}")
    idle, before, waited = wait_for_idle(load, sleep)
    if not idle:
        return _busy(name, before, waited)
    build_suite(SPEED_SUITE, scratch, reference_name=cfg.reference, split="dev", seed=cfg.seed,
                scale=cfg.speed_scale, workers=cfg.workers)
    idle, before, waited = wait_for_idle(load, sleep)
    if not idle:
        return _busy(name, before, waited)
    threads = max(int(result.get("threads") or 1) for result in stored.values())
    run_cfg = RunConfig(suites=(SPEED_SUITE,), tools=(IGBLAST,), threads=threads, igblast_dir=igblast_dir)
    record = run_speed(run_cfg, scratch, truth_reference(scratch, [SPEED_SUITE]), names=list(stored))
    after = load()
    rows, off = {}, []
    for config, old in stored.items():
        new = (record.get("results") or {}).get(config) or {}
        now, then = new.get("steady_reads_per_s"), old["steady_reads_per_s"]
        ratio = None if not now else now / then
        rows[config] = {"stored": then, "now": now, "ratio": ratio, "error": new.get("error"),
                        "stored_utc": old.get("measured_utc"), "same_igblast": new.get("settings") == old.get(
                            "settings") if new.get("settings") else None}
        if ratio is None or abs(ratio - 1.0) > SPEED_BAND:
            off.append(config)
    evidence = {"summary": "; ".join(f"{config} {_rate(row['now'])} vs {_rate(row['stored'])} stored"
                                     + (f" ({100 * (row['ratio'] - 1):+.0f}%)" if row["ratio"] else "")
                                     for config, row in rows.items()),
                "band": SPEED_BAND, "load_before": before, "load_after": after, "waited_seconds": waited,
                "n_full": record.get("n_full"), "repeats": record.get("repeats"), "configs": rows,
                "reads": "a fresh dev S1 (the stored run's S1 is the locked test split)"}
    if off:
        return check(name, "warn", len(rows), evidence,
                     f"informational: {', '.join(off)} outside ± {100 * SPEED_BAND:.0f}% of the stored measurement "
                     "(or not measured); speed never fails the audit")
    return check(name, "pass", len(rows), evidence)


def igblast_v_top1_current(cfg: AuditConfig, sim: RunLayout, speed: RunLayout, scratch: RunLayout) -> dict:
    """``band.sim_v_top1_current``, the gate on IgBLAST's numbers: IgBLAST run now, at this checkout's code, on a fresh
    dev S1 (the one ``band.speed`` built in ``speed`` when it built one, else S1 built at ``cfg.v_top1_scale`` in
    ``scratch``) and scored as ``all`` scores it (against each read's donor genotype); ``pass`` when its
    ``v.top1_in_set`` is within ``V_TOP1_CURRENT_BAND`` of ``V_TOP1_TARGET``, ``fail`` outside it, ``skipped``
    without an IgBLAST release (the audit is then incomplete). ``band.sim_v_top1`` reads the stored test-split run's
    number, computed by older code; this one is computed now, on the dev split. An error raises (``run_audit`` fails
    the check with it)."""
    name = "band.sim_v_top1_current"
    igblast_dir = cfg.igblast_dir or read_run_config(sim).igblast_dir
    if not igblast_dir or not Path(igblast_dir).is_dir():
        return check(name, "skipped", 0, {"summary": f"no IgBLAST release ({igblast_dir})"},
                     f"skipped: no IgBLAST release directory {igblast_dir} (give --igblast-dir)")
    layout = speed if speed.build_json(V_TOP1_SUITE).is_file() else scratch
    if layout is scratch:
        build_suite(V_TOP1_SUITE, scratch, reference_name=cfg.reference, split="dev", seed=cfg.seed,
                    scale=cfg.v_top1_scale, workers=cfg.workers)
    reference = truth_reference(layout, [V_TOP1_SUITE])
    ensure_shared_reference(layout, reference)
    runner = make_runner(IGBLAST, RunnerOptions(igblast_dir=igblast_dir, threads=cfg.threads))
    runner.prepare(layout, reference)
    result = run_suite(runner, layout, V_TOP1_SUITE, arm="shared")
    if result.returncode != 0:
        raise RuntimeError(f"IgBLAST returned {result.returncode}: {result.provenance.get('error')}")
    header, truths = load_truth(layout.truth(V_TOP1_SUITE), header_reference(layout), reference)
    donors = layout.donors(V_TOP1_SUITE)
    scored = score_file(IGBLAST, truths, result.predictions, reference,
                        genotypes=read_genotypes(donors) if donors.is_file() else None)
    values = [scores[V_TOP1_METRIC] for scores in scored.scores.values()
              if math.isfinite(scores.get(V_TOP1_METRIC, math.nan))]
    mean = math.fsum(values) / len(values) if values else None
    evidence = {"summary": f"IgBLAST S1 {V_TOP1_METRIC} {_percent(mean, 2)} (n={len(values)}) on a fresh dev S1 at "
                           f"scale {header.get('scale')}, this code; band {_percent(V_TOP1_TARGET)} ± "
                           f"{100 * V_TOP1_CURRENT_BAND:.1f} pp",
                "mean": mean, "target": V_TOP1_TARGET, "band": V_TOP1_CURRENT_BAND, "split": header.get("split"),
                "scale": header.get("scale"), "cases": len(truths), "settings": runner.fingerprint(),
                "reused_speed_build": layout is speed}
    if mean is None or abs(mean - V_TOP1_TARGET) > V_TOP1_CURRENT_BAND + 1e-12:
        return check(name, "fail", len(values), evidence,
                     f"IgBLAST's S1 {V_TOP1_METRIC} {_percent(mean, 2)} at this code is outside "
                     f"{_percent(V_TOP1_TARGET)} ± {100 * V_TOP1_CURRENT_BAND:.1f} pp: the scoring or the truth "
                     "moved, or IgBLAST did")
    return check(name, "pass", len(values), evidence)


def header_reference(layout: RunLayout) -> str | None:
    """The reference name S1's truth header records."""
    return (read_truth_header(layout.truth(V_TOP1_SUITE)) or {}).get("reference")


def wait_for_idle(load: Callable[[], float], sleep: Callable[[float], None], *, limit: float = SPEED_LOAD,
                  settle_seconds: float = SPEED_SETTLE_SECONDS,
                  poll: float = SPEED_POLL_SECONDS) -> tuple[bool, float, float]:
    """Wait until the 1-minute ``load`` is at most ``limit``, polling every ``poll`` seconds for at most
    ``settle_seconds``: ``(idle, the last load, the seconds waited)``."""
    waited = 0.0
    while True:
        now = float(load())
        if now <= limit:
            return True, now, waited
        if waited >= settle_seconds:
            return False, now, waited
        sleep(poll)
        waited += poll


def _busy(name: str, load: float, waited: float) -> dict:
    return check(name, "skipped", 0, {"summary": f"1-minute load {load:.2f} after {waited:.0f} s of waiting",
                                      "load": load, "waited_seconds": waited},
                 f"skipped (load): the 1-minute load {load:.2f} is above {SPEED_LOAD}")


# --- reproducibility ----------------------------------------------------------------------------------------------

def genairr_digest() -> str | None:
    """The content digest of the GenAIRR source in use, read from its files now (``core.provenance.tree_digest``;
    ``genairr_provenance`` computes it once per process)."""
    source = genairr_provenance().get("genairr_source")
    return tree_digest(source) if source else None


def genairr_identity() -> dict:
    """The GenAIRR in use as a build records it (``workflow.build.GENAIRR_IDENTITY``: version, commit, dirty flag),
    its source digest read now (``genairr_digest``): a commit or an edit changes it."""
    found = genairr_provenance()
    return {**{key: found.get(key) for key in GENAIRR_IDENTITY}, "genairr_source_sha256": genairr_digest()}


def sim_reproducible(cfg: AuditConfig, first: RunLayout, second: RunLayout) -> dict:
    """``reproducible.sim_build``: ``cfg.repro_suite`` built twice on the dev split at ``cfg.repro_scale`` (in
    ``first`` and ``second``); its cases, truth and reads (``REPRO_FILES``, required in both builds) and donors (when
    either build has them) must be byte-identical. ``skipped (GenAIRR
    moved)`` when GenAIRR's source changed before, between or after the builds, or the builds record another
    GenAIRR (the user edits it live): then nothing is compared."""
    name, suite = "reproducible.sim_build", cfg.repro_suite
    digests = [genairr_digest()]
    builds = []
    for layout in (first, second):
        builds.append(build_suite(suite, layout, reference_name=cfg.reference, split="dev", seed=cfg.seed,
                                  scale=cfg.repro_scale, workers=cfg.workers))
        digests.append(genairr_digest())
    recorded = [{key: build["identity"].get(key) for key in GENAIRR_IDENTITY} for build in builds]
    evidence = {"suite": suite, "scale": cfg.repro_scale, "genairr_digests": digests, "genairr_recorded": recorded}
    if len(set(digests)) > 1 or recorded[0] != recorded[1]:
        evidence["summary"] = "GenAIRR's source changed during the two builds; nothing compared"
        return check(name, "skipped", 0, evidence, "skipped (GenAIRR moved): GenAIRR's source changed during the "
                                                   "builds, so they need not match")
    files = {"cases": RunLayout.cases, "truth": RunLayout.truth, "reads": RunLayout.reads, "donors": RunLayout.donors}
    compared, missing = {}, []
    for label, path in files.items():
        a, b = path(first, suite), path(second, suite)
        if label in REPRO_FILES:
            missing += [f"{label} ({which})" for which, found in (("first", a), ("second", b)) if not found.is_file()]
        if not a.is_file() and not b.is_file():
            continue
        compared[label] = {"first": sha256_file(a) if a.is_file() else None,
                           "second": sha256_file(b) if b.is_file() else None}
    differ = [label for label, found in compared.items() if found["first"] != found["second"]]
    n = builds[0]["n_cases"]
    evidence.update(files=compared, missing=missing, identity_same=builds[0]["identity"] == builds[1]["identity"],
                    summary=(f"{suite} at scale {cfg.repro_scale} built twice ({n} cases): "
                             + (f"{', '.join(missing)} missing" if missing else
                                f"{', '.join(differ)} differ" if differ else
                                f"{', '.join(compared)} byte-identical")))
    if missing:
        return check(name, "fail", n, evidence, f"the builds of {suite} are missing {', '.join(missing)}: "
                                                f"{', '.join(REPRO_FILES)} are compared in both builds")
    if differ or n != builds[1]["n_cases"]:
        return check(name, "fail", n, evidence, f"two builds of {suite} on one GenAIRR differ in {', '.join(differ)}: "
                                                "the suite generator or its truth is not deterministic")
    return check(name, "pass", n, evidence)


def jsonl_content(path: Path, aliases: Mapping[str, str] | None = None) -> dict:
    """What a JSON-lines record file holds, by digest: ``file`` (its bytes' SHA-256), ``header`` (the header line's
    record without ``CODE_FIELDS``, each string value found in ``aliases`` replaced by its alias, digested; None
    without a header), ``code`` (the header's ``code_sha256``: the fingerprint also hashes the inputs, which the
    header compares itself), ``rows`` (the other lines, digested) and their count ``n``. A ``.gz`` file is read
    decompressed."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    header = None
    if lines and lines[0].startswith('{"header"'):
        header, lines = json.loads(lines[0])["header"], lines[1:]
    content = None if header is None else _aliased(
        {key: value for key, value in header.items() if key not in CODE_FIELDS}, aliases or {})
    return {"file": sha256_file(path),
            "header": None if content is None else hashlib.sha256(json.dumps(content, sort_keys=True)
                                                                  .encode()).hexdigest(),
            "code": None if header is None else header.get("code_sha256"),
            "rows": hashlib.sha256("\n".join(lines).encode()).hexdigest(), "n": len(lines)}


def _aliased(value, aliases: Mapping[str, str]):
    """``value`` with every string that is a key of ``aliases`` replaced by its alias, in nested dicts and lists."""
    if isinstance(value, dict):
        return {key: _aliased(item, aliases) for key, item in value.items()}
    if isinstance(value, list):
        return [_aliased(item, aliases) for item in value]
    return aliases.get(value, value) if isinstance(value, str) else value


def compare_records(stored: Path, recomputed: Path, *, aliases: Mapping[str, str] | None = None) -> dict:
    """``stored`` against ``recomputed`` (``jsonl_content``): ``identical`` (the content: the rows and the header
    without its code fields), ``file_identical`` (the bytes), ``code_moved`` (the code fields differ), the ``rows``
    count, and ``differs``: what differs (``rows``, ``header``). ``aliases`` maps the byte digest of a stored input
    file to its recomputed twin's, when the two were found to hold the same content: a stored header that names the
    stored file's digest then matches a recomputed one naming its twin's (a file written before JSON-lines gzip was
    reproducible carries a time in its header, so its bytes differ while its content does not)."""
    a, b = jsonl_content(stored, aliases), jsonl_content(recomputed)
    differs = [part for part in ("rows", "header") if a[part] != b[part]]
    return {"identical": not differs, "file_identical": a["file"] == b["file"], "code_moved": a["code"] != b["code"],
            "rows": b["n"], "differs": differs}


def real_reproducible(cfg: AuditConfig, real: RunLayout, scratch: RunLayout) -> dict:
    """``reproducible.real_scores``: ``cfg.real_donor``'s truth (``real_steps.step_truth``), its self-check (the
    pipeline oracle's AIRR TSV and scores included) and every stored run's scores (``real_score.step_scores``)
    recomputed in ``scratch`` from the stored reads, ``reads.tsv``, genotype and predictions, each compared with the
    stored one (``compare_records``; the oracle's TSV byte for byte, the self-check as data). The IUIS map is read from
    the donor's recorded OGRDB file under the data root; the SD3 genes from the stored summary."""
    name, donor = "reproducible.real_scores", cfg.real_donor
    summary = real_summary(real)
    block = (summary.get("donors") or {}).get(donor)
    if block is None or block.get("status") != "done" or donor not in PILOT_SPLIT["dev"]:
        return check(name, "fail", 0, {"summary": f"{donor} is not a scored dev donor of the stored run"},
                     f"{donor} is not a scored dev donor of {real.root}")
    for path in (RunLayout.real_reads, RunLayout.real_reads_meta, RunLayout.real_genotype_json):
        _copy(path(real, donor), path(scratch, donor))
    reference = real_steps.load_reference(summary["config"]["reference"])
    fetched = read_json(real.real_fetch_json(donor)) or {}
    ogrdb = ((fetched.get("records") or {}).get("genotypes") or {}).get("ogrdb") or {}
    iuis = real_steps.load_iuis(Path(cfg.data_root) / ogrdb["path"] if ogrdb.get("path") else None, reference)
    genotype = genotype_from_json((read_json(scratch.real_genotype_json(donor)) or {})["genotype"])
    truths, _, _ = real_steps.step_truth(donor, scratch, genotype, reference, iuis)
    files = {"truth": compare_records(real.real_truth(donor), scratch.real_truth(donor))}
    # the scores' inputs name the truth file by its bytes: the stored truth's digest stands for the recomputed one's
    # when the two hold the same content
    aliases = ({sha256_file(real.real_truth(donor)): sha256_file(scratch.real_truth(donor))}
               if files["truth"]["identical"] else {})
    data = donor_scoring(donor, scratch, truths, genotype, iuis, reference, merge_identical(reference),
                         (summary.get("sensitivity") or {}).get("genes") or ())
    common = scoring_inputs(donor, scratch, iuis, reference)
    redone = self_check(data, scratch, common)
    files[f"shared/{ORACLE}"] = compare_records(real.real_scores(donor, "shared", ORACLE),
                                                scratch.real_scores(donor, "shared", ORACLE), aliases=aliases)
    oracle_tsv = [sha256_file(layout.real_predictions(ORACLE, donor, "shared")) for layout in (real, scratch)]
    self_same = _comparable(redone) == _comparable(block.get("oracle_self_check") or {})
    missing, runs_compared = [], 0
    for arm, runs in (block.get("runs") or {}).items():
        for tool, run in runs.items():
            if run.get("status") != "ok":
                continue
            stored = real.real_scores(donor, arm, tool)
            if not stored.is_file():
                missing.append(f"{arm}/{tool}")  # an ok run is scored; a stored run without its scores is broken
                continue
            _copy(real.real_predictions(tool, donor, arm), scratch.real_predictions(tool, donor, arm))
            recorded = ((read_json(real.real_run_json(tool, donor, arm)) or {}).get("provenance") or {}).get(
                "predictions_sha256")
            real_score.step_scores(data, tool, arm, scratch, {**common, "predictions_sha256": recorded})
            files[f"{arm}/{tool}"] = compare_records(stored, scratch.real_scores(donor, arm, tool), aliases=aliases)
            runs_compared += 1
    differ = [label for label, found in files.items() if not found["identical"]]
    differ += [] if oracle_tsv[0] == oracle_tsv[1] else ["oracle TSV"]
    differ += [] if self_same else ["self-check"]
    differ += [f"{label} (its stored scores are missing)" for label in missing]
    differ += [] if runs_compared else ["no run's scores (no ok run was compared)"]
    moved = [label for label, found in files.items() if found["code_moved"]]
    evidence = {"donor": donor, "files": files, "oracle_tsv_identical": oracle_tsv[0] == oracle_tsv[1],
                "self_check_identical": self_same, "code_moved": moved, "runs_compared": runs_compared,
                "missing_scores": missing,
                "summary": (f"{donor}: truth of {len(truths)} reads, its self-check and {runs_compared} runs' scores "
                            "recomputed: " + (f"{', '.join(differ)} differ" if differ else
                                              "every one identical in content")
                            + (f" (the code computing {', '.join(moved)} moved since: compared without its digest)"
                               if moved else ""))}
    if differ:
        return check(name, "fail", len(truths), evidence,
                     f"recomputed from {donor}'s stored inputs, {', '.join(differ)} differ from the stored records: "
                     "the real truth or scoring is not reproducible, or the stored run is stale for this code")
    return check(name, "pass", len(truths), evidence)


def _comparable(found: dict) -> dict:
    """A self-check as JSON reads it back (tuples as lists, keys as strings)."""
    return json.loads(json.dumps(found, sort_keys=True))


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


# --- null comparisons ---------------------------------------------------------------------------------------------

def compare_problems(comparison: Mapping[str, dict], where: str) -> list[str]:
    """What makes a ``stats.compare`` result (metric -> entry) of one model against itself not null: a Δ other than 0
    (None means no pair; ``pairing`` holds a comparison to its pairs), a verdict other than a tie, a read won or lost,
    each side's own scores differing, or a read one side scores left unpaired (``n_paired`` below either side's own
    ``n``)."""
    problems = []
    for metric, entry in comparison.items():
        delta = entry.get("delta")
        if delta not in (None, 0):
            problems.append(f"{where} {metric}: Δ {delta:+.6g}")
        if entry.get("verdict") not in (None, _TIE):
            problems.append(f"{where} {metric}: verdict {entry['verdict']}")
        if entry.get("wins") or entry.get("losses"):
            problems.append(f"{where} {metric}: {entry.get('wins')} reads won, {entry.get('losses')} lost")
        if entry.get("a") != entry.get("b"):
            problems.append(f"{where} {metric}: each side's own scores differ")
        sides = [(entry.get(side) or {}).get("n") or 0 for side in ("a", "b")]
        if any((entry.get("n_paired") or 0) != n for n in sides):
            problems.append(f"{where} {metric}: pairs {entry.get('n_paired') or 0} of the {sides[0]} and {sides[1]} "
                            "reads the two sides score")
    return problems


def pairing(comparison: Mapping[str, dict], headline: Iterable[str], where: str, *,
            offered: int | None = None) -> tuple[int, list[str]]:
    """What a null comparison rests on, and why it may rest on too little: ``n``, the fewest reads any of the
    ``headline`` metrics pairs (a headline metric not compared pairs none); problems when no metric pairs a read, a
    headline metric was not compared (what is absent is never null), a headline metric pairs no read, or, given the
    reads ``offered`` (the scored cases), the most any headline metric pairs falls short of them (a read never reached
    the comparison). A headline metric that applies to fewer reads (D top-1 where no D is observable) pairs fewer;
    ``compare_problems`` already holds each metric to every read either side scores."""
    problems = [] if _paired(comparison) else [f"{where}: no metric pairs a read"]
    counts = {metric: (comparison[metric].get("n_paired") or 0) if metric in comparison else None
              for metric in headline}
    problems += [f"{where} {metric}: a headline metric, not compared" for metric, n in counts.items() if n is None]
    problems += [f"{where} {metric}: pairs no read" for metric, n in counts.items() if n == 0]
    paired = [n for n in counts.values() if n is not None]
    if offered is not None and paired and max(paired) != offered:
        problems.append(f"{where}: its headline metrics pair at most {max(paired)} of the {offered} reads offered")
    return min((n or 0 for n in counts.values()), default=0), problems


def changes_problems(changes: dict | None, where: str) -> list[str]:
    """What makes ``stats.read_changes`` of one model against itself not null: a read fixed or broken (overall or on a
    metric), or a read only one side scores."""
    if not changes:
        return [f"{where}: no reads fixed and broken were counted"]
    problems = []
    reads = changes.get("reads") or {}
    if reads.get("fixed") or reads.get("broken"):
        problems.append(f"{where}: {reads.get('fixed')} reads fixed, {reads.get('broken')} broken")
    for metric, found in (changes.get("metrics") or {}).items():
        if found.get("fixed") or found.get("broken"):
            problems.append(f"{where} {metric}: {found['fixed']} fixed, {found['broken']} broken")
    if changes.get("a_only") or changes.get("b_only"):
        problems.append(f"{where}: {changes.get('a_only')} reads scored by a only, {changes.get('b_only')} by b only")
    return problems


def _paired(comparison: Mapping[str, dict]) -> int:
    return sum(1 for entry in comparison.values() if entry.get("n_paired"))


def _null_check(name: str, problems: list[str], n: int, evidence: dict, what: str) -> dict:
    evidence["problems"] = problems[:_EXAMPLES]
    evidence["problem_count"] = len(problems)
    if problems:
        return check(name, "fail", n, evidence, f"{what} is not null ({len(problems)} problems, e.g. "
                                                f"{'; '.join(problems[:3])}): the pipeline is non-deterministic or "
                                                "pairs reads unequally")
    return check(name, "pass", n, evidence)


def all_null(cfg: AuditConfig, scratch: RunLayout) -> list[dict]:
    """``null.all`` and ``self_check.sim_oracle_current``: ``all`` with the checkpoint as ``alignair-local@a`` and
    ``@b`` (``workflow.orchestrate.run_all``) on fresh dev builds of ``cfg.null_suites`` at ``cfg.null_scale`` in
    ``scratch``; every comparison must be null (``compare_problems``), and the oracle perfect on every suite.

    ``all`` scores a run only on a build current for the GenAIRR in use, so GenAIRR edited or committed while it ran
    (``genairr_identity`` before and after) leaves its runs stale and unscored: it is run again once (rebuilding what
    moved), and both checks are ``skipped (GenAIRR moved)`` when GenAIRR moves again."""
    tools = tuple(f"alignair-local@{label}" for label in NULL_LABELS)
    run_cfg = RunConfig(suites=tuple(cfg.null_suites), tools=tools, split="dev", reference=cfg.reference,
                        seed=cfg.seed, scale=cfg.null_scale, workers=cfg.workers, threads=cfg.threads,
                        device=cfg.device, checkpoints={label: cfg.checkpoint for label in NULL_LABELS})
    moved = 0
    while True:
        before = genairr_identity()
        returned = run_all(run_cfg, scratch)
        if genairr_identity() == before:
            break
        moved += 1
        if moved > 1:
            evidence = {"summary": "GenAIRR changed while all ran, twice; nothing compared", "genairr_moved": moved}
            reason = "skipped (GenAIRR moved): GenAIRR changed while all ran, so its builds were stale when scored"
            return [check("null.all", "skipped", 0, evidence, reason),
                    check("self_check.sim_oracle_current", "skipped", 0, dict(evidence), reason)]
    report = read_json(scratch.report_json) or {}
    problems = [f"{run['tool']} on {run['suite']}/{run['arm']} {run['status']}: {run.get('error')}"
                for run in report.get("failed_runs") or []]
    built = {suite["name"]: suite for suite in report.get("suites") or []}
    suites, reads = {}, 0
    for name in cfg.null_suites:
        offered = ((built.get(name) or {}).get("cases") or 0) - ((built.get(name) or {}).get("excluded") or 0)
        for arm in get_suite(name).arms:
            found = read_json(scratch.comparison_json(name, arm))
            if found is None:
                problems.append(f"{name}/{arm}: not compared")
                continue
            rows = found.get("comparison") or {}
            problems += compare_problems(rows, f"{name}/{arm}")
            n, short = pairing(rows, SIM_HEADLINE, f"{name}/{arm}", offered=offered)
            problems += short
            problems += [f"{name}/{arm} {metric}: the informational summaries differ"
                         for metric, sides in (found.get("informational") or {}).items()
                         if sides.get("a") != sides.get("b")]
            slices = found.get("by_slice") or {}
            if slices.get("a") != slices.get("b"):
                problems.append(f"{name}/{arm}: the two tools' per-slice summaries differ")
            pair = [scratch.predictions(tool, name, arm) for tool in tools]
            suites[f"{name}/{arm}"] = {"metrics": len(rows), "paired_metrics": _paired(rows), "n": n,
                                       "offered": offered,
                                       "predictions_identical": all(path.is_file() for path in pair)
                                       and sha256_file(pair[0]) == sha256_file(pair[1])}
            reads += n if arm == "shared" else 0
    runs = {}
    for tool, suite, arm in dict.fromkeys([(ORACLE, name, "shared") for name in cfg.null_suites]
                                          + [(tool, name, arm) for tool in tools for name in cfg.null_suites
                                             for arm in get_suite(name).arms]):
        try:
            runs[f"{tool} {suite}/{arm}"] = read_run_json(scratch.run_json(tool, suite, arm)).returncode
        except (OSError, ValueError, TypeError):
            runs[f"{tool} {suite}/{arm}"] = None
    problems += [f"{run}: {'no run record' if code is None else f'return code {code}'}"
                 for run, code in runs.items() if code != 0]
    evidence = {"summary": ", ".join(f"{key}: {entry['metrics']} metrics, every headline metric pairing at least "
                                     f"{entry['n']} of {entry['offered']} reads, predictions "
                                     f"{'byte-identical' if entry['predictions_identical'] else 'differ'}"
                                     for key, entry in suites.items()) or "nothing compared",
                "tools": list(tools), "scale": cfg.null_scale, "returncode": returned, "suites": suites,
                "runs": runs, "genairr_moved": moved}
    null = _null_check("null.all", problems, reads, evidence, "all's comparison of one checkpoint with itself")
    oracle = {suite["name"]: suite.get("oracle") or {} for suite in report.get("suites") or []}
    bad = [name for name in cfg.null_suites if (oracle.get(name) or {}).get("status") != "perfect"]
    cases = sum((oracle.get(name) or {}).get("cases") or 0 for name in cfg.null_suites)
    current = {"summary": f"oracle perfect on {len(cfg.null_suites) - len(bad)} of {len(cfg.null_suites)} fresh dev "
                          f"suites ({cases} cases) at this checkout's code",
               "suites": {name: oracle.get(name) for name in cfg.null_suites}}
    if bad:
        return [null, check("self_check.sim_oracle_current", "fail", cases, current,
                            f"the oracle is not perfect on the fresh dev {', '.join(bad)}")]
    return [null, check("self_check.sim_oracle_current", "pass", cases, current)]


def diagnose_null(cfg: AuditConfig, scratch: RunLayout) -> dict:
    """``null.diagnose``: fresh dev builds of ``cfg.diagnose_suites`` in ``scratch``, then the checkpoint file
    diagnosed against itself loaded in memory (``workflow.diagnose.evaluate(..., against=(model,
    reference_set))``): two model keys, each diagnosed on the same reads, whose comparison must be null. One model
    key (one record compared with itself, zero by construction) fails, and so does a comparison missing anything it
    must hold: a suite of ``cfg.diagnose_suites`` (or one more), a suite's head family or a boundary metric of each
    gene and edge in it, its end-to-end family or a headline metric in it, or the suite's scored reads
    (``reads.scored``, what the headline metrics must pair)."""
    # torch and alignair are imported only when diagnostics run
    from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS
    from alignair_assay.diagnostics.replay import load_model_checked
    from alignair_assay.workflow.diagnose import evaluate, parse_device

    names = list(cfg.diagnose_suites)
    for suite in names:
        build_suite(suite, scratch, reference_name=cfg.reference, split="dev", seed=cfg.seed,
                    scale=cfg.diagnose_scale, workers=cfg.workers)
    device = parse_device(cfg.device, fix="--device cpu")
    in_memory = load_model_checked(cfg.checkpoint, truth_reference(scratch, names), device=device)
    variants = DEFAULT_VARIANTS if cfg.diagnose_variants is None else cfg.diagnose_variants
    result = evaluate(cfg.checkpoint, scratch, names, device=device, variants=variants, profile=False,
                      against=in_memory)
    against = result.against
    problems = []
    if against["same_model"] or against["baseline"] == result.model_key:
        problems.append("one model key: the file was compared with its own record, zero by construction")
    compared = list(against.get("suites") or {})
    if sorted(compared) != sorted(names):
        problems.append(f"the comparison holds the suites {', '.join(compared) or 'none'}, not the diagnosed "
                        f"{', '.join(names)}")
    suites, reads = {}, 0
    for name, suite in (against.get("suites") or {}).items():
        offered = (((getattr(result, "suites", None) or {}).get(name) or {}).get("reads") or {}).get("scored")
        if offered is None:
            problems.append(f"{name}: the diagnosis records no scored reads (reads.scored)")
        entry = {"cases": suite.get("cases"), "offered": offered}
        for family in ("heads", "baseline"):
            found = suite.get(family)
            if not found:
                entry[family] = None
                continue
            where = f"{name} {family}"
            problems += compare_problems(found["comparison"], where)
            problems += changes_problems(found.get("changes"), where)
            entry[family] = {"metrics": len(found["comparison"]), "paired_metrics": _paired(found["comparison"]),
                             "fixed": found["changes"]["reads"]["fixed"], "broken": found["changes"]["reads"]["broken"]}
            if family == "baseline":
                entry[family]["n"], short = pairing(found["comparison"], SIM_HEADLINE, where, offered=offered)
                problems += short
            else:
                problems += [] if entry[family]["paired_metrics"] else [f"{where}: no metric pairs a read"]
                problems += [f"{where}: no {prefix}* boundary metric" for prefix in HEAD_BOUNDARY_PREFIXES
                             if not any(metric.startswith(prefix) for metric in found["comparison"])]
                problems += [f"{where} {metric}: pairs no read" for metric, row in found["comparison"].items()
                             if metric.startswith(HEAD_BOUNDARY_PREFIXES) and not row.get("n_paired")]
        if entry["heads"] is None:
            problems.append(f"{name}: no head family was compared")
        if entry["baseline"] is None:
            problems.append(f"{name}: no end-to-end family was compared (no variant ran)")
        reads += (entry["baseline"] or {}).get("n") or 0
        suites[name] = entry
    shown = []
    for name, entry in suites.items():
        families = [f"{family} {found['metrics']} metrics" for family, found in entry.items()
                    if isinstance(found, dict)]
        least = (entry["baseline"] or {}).get("n")
        shown.append(f"{name} {entry['cases']} reads ({', '.join(families)}; headline metrics pair at least {least})")
    evidence = {"summary": f"file {result.model_key} against in-memory {against['baseline']}: {', '.join(shown)}",
                "model_key": result.model_key, "baseline": against["baseline"], "device": device,
                "variants": list(variants), "suites": suites}
    return _null_check("null.diagnose", problems, reads, evidence,
                       "diagnose's comparison of the checkpoint with itself in memory")


def real_null(cfg: AuditConfig, scratch: RunLayout) -> dict:
    """``null.real``: ``real`` with the checkpoint as ``alignair-api@a`` and ``@b`` on ``cfg.real_donor``, capped at
    ``cfg.real_cap`` (``workflow.real.run_real``) in ``scratch``. Every checkpoint comparison (per donor and pooled,
    each arm: overall, per stratum, the sensitivity analyses and the clusterings) must be null, with no read fixed
    or broken, equal one-sided exclusions and informational summaries, and the two runs' own summaries equal."""
    tools = tuple(f"alignair-api@{label}" for label in NULL_LABELS)
    real_cfg = RealConfig(donors=(cfg.real_donor,), split="dev", data_root=cfg.data_root, cap=cfg.real_cap,
                          tools=tools, threads=cfg.threads, device=cfg.device, reference=cfg.reference,
                          checkpoints={label: cfg.checkpoint for label in NULL_LABELS})
    summary = run_real(real_cfg, scratch)
    problems = [f"{stopped['donor']} stopped at {stopped['step']}: {stopped['reason']}"
                for stopped in summary.get("stopped_donors") or []]
    problems += [f"{failed['tool']} on {failed['donor']}'s {failed['arm']} arm failed: {failed['error']}"
                 for failed in summary.get("failed_runs") or []]
    block = (summary.get("donors") or {}).get(cfg.real_donor) or {}
    arms, reads = {}, 0
    comparisons = [(f"{cfg.real_donor}", entry) for entry in block.get("comparisons") or []]
    comparisons += [("pooled", entry) for entry in (summary.get("pooled") or {}).get("comparisons") or []]
    for where, entry in comparisons:
        if entry.get("kind") != "checkpoint":
            continue
        label = f"{where} {entry['arm']}"
        if entry.get("status") != "compared":
            problems.append(f"{label}: not compared ({entry.get('reason')})")
            continue
        problems += real_comparison_problems(entry, label)
        n, short = pairing(entry.get("overall") or {}, REAL_HEADLINE, f"{label} overall")
        problems += short
        if where != "pooled":
            arms[entry["arm"]] = {"reads": entry.get("n_reads"), "n": n, "metrics": len(entry.get("overall") or {}),
                                  "paired_metrics": _paired(entry.get("overall") or {}),
                                  "fixed": ((entry.get("changes") or {}).get("reads") or {}).get("fixed"),
                                  "broken": ((entry.get("changes") or {}).get("reads") or {}).get("broken")}
            reads += n if entry["arm"] == "shared" else 0
    for arm, runs in (block.get("tools") or {}).items():
        first, second = (runs.get(tool) for tool in tools)
        if first is not None and first != second:
            problems.append(f"{cfg.real_donor} {arm}: the two runs' own R1 summaries differ")
    for arm in ("shared", "genotype"):
        if arm not in arms:
            problems.append(f"{cfg.real_donor} {arm}: no checkpoint comparison")
        else:
            pair = [scratch.real_predictions(tool, cfg.real_donor, arm) for tool in tools]
            arms[arm]["predictions_identical"] = (all(path.is_file() for path in pair)
                                                  and sha256_file(pair[0]) == sha256_file(pair[1]))
    oracle = (block.get("oracle_self_check") or {}).get("status")
    evidence = {"summary": f"{cfg.real_donor} capped at {cfg.real_cap}: " + ", ".join(
                    f"{arm} {entry['metrics']} metrics over {entry['reads']} reads (headline metrics pair at least "
                    f"{entry['n']}), fixed {entry['fixed']} / broken {entry['broken']}" for arm, entry in arms.items())
                + f"; its self-check {oracle}",
                "donor": cfg.real_donor, "cap": cfg.real_cap, "tools": list(tools), "arms": arms,
                "oracle_self_check": oracle}
    return _null_check("null.real", problems, reads, evidence, "real's comparison of one checkpoint with itself")


def real_comparison_problems(entry: dict, where: str) -> list[str]:
    """``compare_problems`` over every comparison of one ``real`` pair (overall, per stratum, the sensitivity
    analyses, the clusterings), ``changes_problems`` of its reads fixed and broken, and the two runs' one-sided
    exclusions and informational summaries, which must be equal."""
    problems = compare_problems(entry.get("overall") or {}, f"{where} overall")
    for label, found in (entry.get("strata") or {}).items():
        problems += compare_problems(found, f"{where} {label}")
    for label, found in (entry.get("sensitivity") or {}).items():
        problems += compare_problems(found.get("comparison") or {}, f"{where} sensitivity {label}")
    for label, found in (entry.get("clusterings") or {}).items():
        problems += compare_problems(found.get("comparison") or {}, f"{where} clustering {label}")
    problems += changes_problems(entry.get("changes"), where)
    one_sided = entry.get("one_sided") or {}
    if one_sided.get("a") != one_sided.get("b"):
        problems.append(f"{where}: the two runs exclude other reads on their own")
    for metric, found in (entry.get("informational") or {}).items():
        if found.get("a") != found.get("b"):
            problems.append(f"{where} {metric}: the informational summaries differ")
    return problems


# --- the audit ----------------------------------------------------------------------------------------------------

def check_inputs(cfg: AuditConfig, layout: RunLayout) -> None:
    """Refusals before any work: FileNotFoundError for a stored run without its records (an ``all`` run's
    ``manifest.json`` and ``report.json``, a ``real`` run's summary), a missing data root or checkpoint; ValueError
    for an output directory that overlaps either stored run (they are read only) or a CUDA device torch does not
    see."""
    sim, real = RunLayout(cfg.sim_run), RunLayout(cfg.real_run)
    for path in (sim.manifest, sim.report_json):
        if not path.is_file():
            raise FileNotFoundError(f"no {path}: --sim-run is a run directory made by `all`")
    if not real.real_summary_json().is_file():
        raise FileNotFoundError(f"no {real.real_summary_json()}: --real-run is a run directory made by `real`")
    if not Path(cfg.data_root).is_dir():
        raise FileNotFoundError(f"no data root {cfg.data_root} (--data-root)")
    if not Path(cfg.checkpoint).is_file():
        raise FileNotFoundError(f"no AlignAIR checkpoint {cfg.checkpoint} (--checkpoint {CHECKPOINT_LABEL}=PATH)")
    out = Path(layout.root).resolve()
    for run in (sim.root, real.root):
        run = Path(run).resolve()
        if out == run or run in out.parents or out in run.parents:
            raise ValueError(f"the audit's output {out} overlaps the stored run {run}, which it only reads: choose "
                             "another --out")
    if cfg.device != "cpu":
        from alignair_assay.workflow.diagnose import parse_device  # imports torch

        parse_device(cfg.device, fix="--device cpu")


def run_audit(cfg: AuditConfig, layout: RunLayout) -> dict:
    """Run every check (the module docstring) and write ``layout.audit_json`` and ``layout.audit_md``; returns the
    record: ``kind``, ``version``, the ``verdict``, the ``config``, the audited ``checkpoint`` (path and SHA-256),
    the ``provenance`` (host, the assay's git state and ``audit_code_sha256``, the digest of this whole package's
    sources, both taken before any check runs, so the record names the code it started with; the GenAIRR in use),
    ``created_utc``, ``seconds`` and the ``checks`` in ``CHECKS`` order, each printed as it finishes. IgBLAST is timed
    first, while the machine is quietest. A check that raises fails with its error as the reason (an
    ``INFORMATIONAL`` one warns), and the others go on. Refusals (``check_inputs``) come before any work."""
    start = time.monotonic()
    assay, code = git_state(_PACKAGE), tree_digest(_PACKAGE, suffixes=CODE_SUFFIXES)
    cfg = dataclasses.replace(cfg, sim_run=absolute_path(cfg.sim_run), real_run=absolute_path(cfg.real_run),
                              data_root=absolute_path(cfg.data_root), checkpoint=absolute_path(cfg.checkpoint),
                              igblast_dir=absolute_path(cfg.igblast_dir) if cfg.igblast_dir else None)
    layout = RunLayout(absolute_path(layout.root))
    check_inputs(cfg, layout)
    sim, real = RunLayout(cfg.sim_run), RunLayout(cfg.real_run)
    layout.audit_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}

    def scratch(name: str) -> RunLayout:
        path = layout.audit_scratch(name)
        shutil.rmtree(path, ignore_errors=True)
        return RunLayout(path)

    speed = scratch("speed")
    steps: list[tuple[tuple[str, ...], Callable[[], dict | list[dict]]]] = [
        (("band.speed",), lambda: igblast_speed(cfg, sim, speed)),
        (("band.sim_v_top1_current",),
         lambda: igblast_v_top1_current(cfg, sim, speed, scratch("sim_v_top1_current"))),
        (("self_check.sim_oracle",), lambda: sim_oracle(sim)),
        (("self_check.sim_staleness",), lambda: sim_staleness(sim)),
        (("self_check.real_oracle",), lambda: real_oracle(real)),
        (("self_check.real_truth_health",), lambda: real_truth_health(real)),
        (("band.real_known",), lambda: real_known_band(real)),
        (("band.sim_v_top1",), lambda: sim_v_top1_band(sim)),
        (("reproducible.sim_build",), lambda: sim_reproducible(cfg, scratch("sim_build_1"), scratch("sim_build_2"))),
        (("reproducible.real_scores",), lambda: real_reproducible(cfg, real, scratch("real_scores"))),
        (("null.all", "self_check.sim_oracle_current"), lambda: all_null(cfg, scratch("null_all"))),
        (("null.diagnose",), lambda: diagnose_null(cfg, scratch("null_diagnose"))),
        (("null.real",), lambda: real_null(cfg, scratch("null_real"))),
    ]
    for names, step in steps:
        started = time.monotonic()
        try:
            results = step()
        except Exception as error:  # a check that raises fails (an informational one warns); the audit goes on
            results = [check(name, "warn" if name in INFORMATIONAL else "fail", 0,
                             {"summary": "raised before it could check"},
                             ("informational: " if name in INFORMATIONAL else "") + f"{type(error).__name__}: {error}")
                       for name in names]
        for result in results if isinstance(results, list) else [results]:
            result = settle(result)
            result["evidence"]["seconds"] = round(time.monotonic() - started, 1)
            found[result["name"]] = result
            print(f"audit {result['name']}: {result['status']} (n={result['n']}) "
                  f"{result['evidence'].get('summary') or ''}" + (f"; {result['reason']}" if result["reason"] and
                                                                  result["status"] != "pass" else ""), flush=True)
    checks = [found[name] for name in CHECKS if name in found]
    genairr = genairr_provenance()
    record = {"kind": AUDIT_KIND, "version": AUDIT_VERSION, "verdict": verdict(checks),
              "config": dataclasses.asdict(cfg),
              "checkpoint": {"label": CHECKPOINT_LABEL, "path": cfg.checkpoint, "sha256": sha256_file(cfg.checkpoint)},
              "gating": list(GATING),
              "provenance": {"host": platform.node(), "assay": assay, "audit_code_sha256": code,
                             "genairr": genairr, "genairr_freeze": frozen_record(genairr.get("genairr_source"))},
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "seconds": round(time.monotonic() - start, 1), "checks": checks, "groups": GROUPS}
    record = json.loads(json.dumps(record, default=_jsonable))
    write_json(layout.audit_json, record)
    write_text(layout.audit_md, audit_markdown(record))
    return record


def _jsonable(value):
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON: {type(value).__name__}")


def _percent(value, digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def _rate(value) -> str:
    return "—" if not value else f"{value:.1f} reads/s"
