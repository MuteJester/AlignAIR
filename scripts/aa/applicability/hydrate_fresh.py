"""Stage-1 hydration orchestration + GO/STOP gate for the fresh validation cohort.

This module provides the *pure orchestration* over already-collected fresh records:
disjointness audit (never touches the seal) -> never-pooled selection/blind-LOCO allocation ->
guardband + VHH-minimum gate -> private cohort/allocation + committed counts-only census.

The multi-source network hydration itself (OAS / VDJServer / UniProt / SAbDab / GenAIRR / structural
transforms, each pulled disjoint from the verified pre-split Gate-3 denylist) is performed by the
supervised Stage-1 execution harness, which calls ``run_stage1`` with the collected records. The locked
partition file is never read.
"""
from __future__ import annotations

import json
from pathlib import Path

from alignair.aa.guard.allocate import allocate_fresh
from alignair.aa.guard.disjoint import audit_disjoint


def stage1_decision(status: dict, clean: bool) -> str:
    """Pure Stage-1 gate. GO iff every group meets its guardband floor, both VHH minimums hold, and the
    disjointness audit is clean. Otherwise STOP_FRESH_COHORT_UNPOWERED."""
    floors_ok = all(status.get("floors_ok", {}).values()) and bool(status.get("floors_ok"))
    vhh_ok = all(status.get("vhh_min_ok", {}).values()) and bool(status.get("vhh_min_ok"))
    if clean and floors_ok and vhh_ok:
        return "GO"
    reasons = []
    if not clean:
        reasons.append("disjointness_overlap")
    if not floors_ok:
        reasons.append("guardband_floor")
    if not vhh_ok:
        reasons.append("vhh_minimum")
    return "STOP_FRESH_COHORT_UNPOWERED:" + ",".join(reasons)


def _counts(alloc) -> dict:
    return {
        g: {"selection": len(alloc.selection.get(g, set())),
            "loco_challenge": len(alloc.loco_challenge.get(g, set()))}
        for g in sorted(set(alloc.selection) | set(alloc.loco_challenge))
    }


def run_stage1(fresh_records, *, gate3_denylist, phase3, vhh_temporal_cutoff,
               private_dir, artifact_dir) -> str:
    """Audit disjointness, allocate, decide, and write the private cohort + committed counts-only census.

    Returns the Stage-1 decision string. Writes:
      - `<private_dir>/fresh_cohort.jsonl`, `<private_dir>/allocation.json` (lineage keys; git-ignored)
      - `<artifact_dir>/fresh_cohort_census.json` (counts only; committable)
    """
    private_dir = Path(private_dir)
    artifact_dir = Path(artifact_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report = audit_disjoint(fresh_records, gate3_denylist=gate3_denylist, phase3=phase3)

    census: dict = {
        "artifact": "fresh_cohort_census",
        "disjoint_clean": report.clean,
        "overlap_counts": {k: len(v) for k, v in report.overlaps.items()},
    }

    if not report.clean:
        census["decision"] = stage1_decision({}, clean=False)
        (artifact_dir / "fresh_cohort_census.json").write_text(json.dumps(census, indent=2, sort_keys=True))
        return census["decision"]

    alloc = allocate_fresh(fresh_records, vhh_temporal_cutoff=vhh_temporal_cutoff)
    status = {"floors_ok": alloc.floors_ok, "vhh_min_ok": alloc.vhh_min_ok}
    decision = stage1_decision(status, clean=report.clean)

    # Private allocation (lineage keys) + fresh cohort (sequences) — git-ignored.
    (private_dir / "allocation.json").write_text(json.dumps({
        "selection": {g: sorted(s) for g, s in alloc.selection.items()},
        "loco_challenge": {g: sorted(s) for g, s in alloc.loco_challenge.items()},
        "vhh_mode": alloc.vhh_mode,
    }, indent=2, sort_keys=True))
    with open(private_dir / "fresh_cohort.jsonl", "w") as fh:
        for r in fresh_records:
            fh.write(json.dumps(r) + "\n")

    # Committed census — counts only, no sequences.
    census.update({
        "decision": decision,
        "counts": _counts(alloc),
        "floors_ok": alloc.floors_ok,
        "vhh_min_ok": alloc.vhh_min_ok,
        "vhh_mode": alloc.vhh_mode,
        "pooled_overlap": len(alloc.pooled_overlap),
    })
    (artifact_dir / "fresh_cohort_census.json").write_text(json.dumps(census, indent=2, sort_keys=True))
    return decision
