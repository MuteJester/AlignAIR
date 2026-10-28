"""The audit itself: its refusals before any work, every check in order (each in a scratch run directory of its own),
and its record."""
from __future__ import annotations

import dataclasses
import json
import platform
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.freeze import frozen_record
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import CODE_SUFFIXES, genairr_provenance, git_state, sha256_file, tree_digest
from alignair_assay.core.records import write_json, write_text
from alignair_assay.report.audit import audit_markdown
from alignair_assay.runners.base import absolute_path
from alignair_assay.workflow.audit import igblast, nulls, reproducible, stored
from alignair_assay.workflow.audit.base import (
    AUDIT_KIND,
    AUDIT_VERSION,
    AuditConfig,
    CHECKPOINT_LABEL,
    CHECKS,
    GATING,
    GROUPS,
    INFORMATIONAL,
    _PACKAGE,
    check,
    settle,
    verdict,
)

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
    """Run every check (the ``workflow.audit`` docstring) and write ``layout.audit_json`` and ``layout.audit_md``;
    returns the record: ``kind``, ``version``, the ``verdict``, the ``config``, the audited ``checkpoint`` (path and
    SHA-256), the ``provenance`` (host, the assay's git state and ``audit_code_sha256``, the digest of the whole
    ``alignair_assay`` package's sources, both taken before any check runs, so the record names the code it started
    with; the GenAIRR in use), ``created_utc``, ``seconds`` and the ``checks`` in ``CHECKS`` order, each printed as it
    finishes. IgBLAST is timed first, while the machine is quietest. A check that raises fails with its error as the
    reason (an ``INFORMATIONAL`` one warns), and the others go on; a ``GATING`` check no step returned is skipped,
    named in its reason (the verdict is then incomplete). Refusals (``check_inputs``) come before any work."""
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

    speed, null_sim = scratch("speed"), scratch("null_sim")
    steps: list[tuple[tuple[str, ...], Callable[[], dict | list[dict]]]] = [
        (("band.speed",), lambda: igblast.igblast_speed(cfg, sim, speed)),
        (("band.sim_v_top1_current",),
         lambda: igblast.igblast_v_top1_current(cfg, sim, speed, scratch("sim_v_top1_current"))),
        (("self_check.sim_oracle",), lambda: stored.sim_oracle(sim)),
        (("self_check.sim_staleness",), lambda: stored.sim_staleness(sim)),
        (("self_check.real_oracle",), lambda: stored.real_oracle(real)),
        (("self_check.real_truth_health",), lambda: stored.real_truth_health(real)),
        (("band.real_known",), lambda: igblast.real_known_band(real)),
        (("band.sim_v_top1",), lambda: igblast.sim_v_top1_band(sim)),
        (("reproducible.sim_build",),
         lambda: reproducible.sim_reproducible(cfg, scratch("sim_build_1"), scratch("sim_build_2"))),
        (("reproducible.real_scores",), lambda: reproducible.real_reproducible(cfg, real, scratch("real_scores"))),
        (("null.all", "self_check.sim_oracle_current"), lambda: nulls.all_null(cfg, null_sim)),
        (("null.diagnose",), lambda: nulls.diagnose_null(cfg, null_sim)),
        (("positive.diagnose",), lambda: nulls.positive_diagnose(cfg, null_sim, scratch("positive"))),
        (("agreement.all_vs_diagnose",), lambda: nulls.all_vs_diagnose(cfg, null_sim)),
        (("null.real",), lambda: nulls.real_null(cfg, scratch("null_real"))),
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
    for name in GATING:
        if name not in found:  # a gating check no step returned was not checked, so it may have failed
            found[name] = check(name, "skipped", 0, {"summary": "no result"},
                                f"skipped: no step of the audit returned a result for {name}")
            print(f"audit {name}: skipped (n=0) {found[name]['reason']}", flush=True)
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
