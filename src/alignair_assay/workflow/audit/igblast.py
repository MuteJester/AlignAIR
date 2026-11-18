"""IgBLAST against independent evidence: its known answers on the stored real run and its S1 V top-1 in the stored
simulated run (read only), and, live, its S1 V top-1 at this code and its speed."""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.records import read_json
from alignair_assay.scoring.pipeline import read_scores, truth_reference
from alignair_assay.truth.io import read_truth_header
from alignair_assay.workflow.build import build_suite
from alignair_assay.workflow.orchestrate import RunConfig, read_run_config, run_all
from alignair_assay.workflow.speed import run_speed
from alignair_assay.workflow.summary import ORACLE, configured_settings
from alignair_assay.workflow.audit.base import (
    AuditConfig,
    IGBLAST,
    KNOWN_FLOOR,
    KNOWN_METRICS,
    SPEED_BAND,
    SPEED_LOAD,
    SPEED_POLL_SECONDS,
    SPEED_SETTLE_SECONDS,
    SPEED_SUITE,
    V_TOP1_BAND,
    V_TOP1_CURRENT_BAND,
    V_TOP1_METRIC,
    V_TOP1_SUITE,
    V_TOP1_TARGET,
    _percent,
    _rate,
    check,
)
from alignair_assay.workflow.audit.stored import _missing, dev_donors, real_summary

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
    ``scratch``) through ``all``'s own path (``workflow.orchestrate.run_all`` with the oracle and IgBLAST, so it is
    run and scored as ``all`` runs and scores it: against each read's donor genotype); ``pass`` when its
    ``v.top1_in_set`` is within ``V_TOP1_CURRENT_BAND`` of ``V_TOP1_TARGET``, ``fail`` outside it, ``skipped``
    without an IgBLAST release (the audit is then incomplete). ``band.sim_v_top1`` reads the stored test-split run's
    number, computed by older code; this one is computed now, on the dev split. An error, a failed run or an
    imperfect oracle raises (``run_audit`` fails the check with it)."""
    name = "band.sim_v_top1_current"
    igblast_dir = cfg.igblast_dir or read_run_config(sim).igblast_dir
    if not igblast_dir or not Path(igblast_dir).is_dir():
        return check(name, "skipped", 0, {"summary": f"no IgBLAST release ({igblast_dir})"},
                     f"skipped: no IgBLAST release directory {igblast_dir} (give --igblast-dir)")
    layout = speed if speed.build_json(V_TOP1_SUITE).is_file() else scratch
    if layout is scratch:
        build_suite(V_TOP1_SUITE, scratch, reference_name=cfg.reference, split="dev", seed=cfg.seed,
                    scale=cfg.v_top1_scale, workers=cfg.workers)
    build = read_json(layout.build_json(V_TOP1_SUITE)) or {}
    run_cfg = RunConfig(suites=(V_TOP1_SUITE,), tools=(ORACLE, IGBLAST), split="dev", reference=cfg.reference,
                        seed=build.get("seed", cfg.seed), scale=build.get("scale", cfg.v_top1_scale),
                        workers=cfg.workers, threads=cfg.threads, igblast_dir=igblast_dir)
    returned = run_all(run_cfg, layout, force=True)
    failed = (read_json(layout.report_json) or {}).get("failed_runs") or []
    if returned != 0 or failed:
        raise RuntimeError(f"all of IgBLAST on {V_TOP1_SUITE} returned {returned}: "
                           + ("; ".join(f"{run['tool']} {run['status']}: {run.get('error')}" for run in failed)
                              or "the oracle is not perfect"))
    header = read_truth_header(layout.truth(V_TOP1_SUITE)) or {}
    scores = read_scores(layout.tool_scores(V_TOP1_SUITE, "shared", IGBLAST))
    values = [case[V_TOP1_METRIC] for case in scores.values() if math.isfinite(case.get(V_TOP1_METRIC, math.nan))]
    mean = math.fsum(values) / len(values) if values else None
    evidence = {"summary": f"IgBLAST S1 {V_TOP1_METRIC} {_percent(mean, 2)} (n={len(values)}) on a fresh dev S1 at "
                           f"scale {header.get('scale')}, this code; band {_percent(V_TOP1_TARGET)} ± "
                           f"{100 * V_TOP1_CURRENT_BAND:.1f} pp",
                "mean": mean, "target": V_TOP1_TARGET, "band": V_TOP1_CURRENT_BAND, "split": header.get("split"),
                "scale": header.get("scale"), "cases": len(scores),
                "settings": configured_settings(layout).get(IGBLAST), "reused_speed_build": layout is speed}
    if mean is None or abs(mean - V_TOP1_TARGET) > V_TOP1_CURRENT_BAND + 1e-12:
        return check(name, "fail", len(values), evidence,
                     f"IgBLAST's S1 {V_TOP1_METRIC} {_percent(mean, 2)} at this code is outside "
                     f"{_percent(V_TOP1_TARGET)} ± {100 * V_TOP1_CURRENT_BAND:.1f} pp: the scoring or the truth "
                     "moved, or IgBLAST did")
    return check(name, "pass", len(values), evidence)


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
