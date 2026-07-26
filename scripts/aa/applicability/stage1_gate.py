"""Final Stage-1 gate: assemble all fresh sources, allocate (study/donor + genus + lineage), derive
structural negatives from the finalized real-IGH parents, globally dedup AFTER assignment, and run the
owner-required audit -> Stage-1 GO/STOP.

Order (per owner directive): real sources -> combine_allocation (sets positive_real roles) -> structural
inherits each real-IGH parent's role+lineage -> global sequence dedup -> re-check floors -> audit.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/stage1_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

from alignair.aa.guard.stage1_audit import audit_stage1, combine_allocation, global_dedup
from alignair.aa.guard.structural import structural_records

STORE = Path(".private/aa-applicability/fresh_hydration")
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability")
CENSUS_DIR = ART / "fresh_hydration"

SOURCES = ["sabdab", "igsf", "genairr_sim", "oas", "vdjserver"]


def _load_sources():
    records, present, checksums = [], [], {}
    for s in SOURCES:
        p = STORE / f"{s}.normalized.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text())
        records.extend(recs)
        present.append(f"{s}({len(recs)})")
        cp = CENSUS_DIR / f"{s}.census.json"
        if cp.exists():
            prov = json.loads(cp.read_text()).get("provenance", {})
            checksums[s] = prov.get("raw_sha256") or prov.get("cache_sha256") or prov.get("uniprot_raw_sha256") or "n/a"
    return records, present, checksums


def main():
    ART.mkdir(parents=True, exist_ok=True)
    records, present, checksums = _load_sources()

    selection, loco, study_alloc = combine_allocation(records)

    # structural inherits each real-IGH parent's role + lineage (generated LAST, from real IGH only)
    real = [r for r in records if r["group"] == "positive_real"]
    role_of_lineage = {}
    for lk in selection.get("positive_real", set()):
        role_of_lineage[lk] = "selection"
    for lk in loco.get("positive_real", set()):
        role_of_lineage[lk] = "loco"
    structural = []
    for parent in real:
        role = role_of_lineage.get(parent["lineage_key"])
        if role is None:
            continue
        structural.extend(structural_records(parent, role_of_parent=role))
    for r in structural:
        (selection if r["inherited_role"] == "selection" else loco).setdefault("structural", set()).add(r["lineage_key"])

    all_records = records + structural

    # global sequence dedup AFTER assignment, then re-check floors in the audit
    kept, dropped = global_dedup(all_records)

    report = audit_stage1(records=kept, selection=selection, loco=loco, study_alloc=study_alloc,
                          source_checksums=checksums)

    census = {
        "artifact": "stage1_gate",
        "sources_present": present,
        "source_checksums": checksums,
        "total_records": len(all_records),
        "global_dedup_dropped": dropped,
        "kept": len(kept),
        "decision": report.decision,
        "role_overlap": report.role_overlap,
        "counts_by_role": report.counts_by_role,
        "counts_by_locus": report.counts_by_locus,
        "counts_by_study": report.counts_by_study,
        "counts_by_donor": report.counts_by_donor,
        "floors_ok": report.floors_ok,
        "locus_balance_37_37": report.locus_balance_37_37,
        "powered_real_igh": report.powered_real_igh,
        "policy_hash_reproduced": report.policy_hash_reproduced,
        "allocator_policy_hash": study_alloc.policy_hash if study_alloc else None,
        "failures": report.failures,
    }
    (ART / "STAGE1_RESULT.json").write_text(json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))
    return report.decision


if __name__ == "__main__":
    main()
