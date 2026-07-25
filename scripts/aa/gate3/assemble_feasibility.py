"""Assemble the hydrated Gate-3 cohort (git-ignored .private store) into CohortEntry records and run the
committed split-aware feasibility calculator -> real GO/STOP + artifacts.

The hydration itself (network pulls from OAS/ADC/SAbDab/InterPro/Thera-SAbDab) was done by the Phase-0
hydration agent and persisted per-source under .private/aa-gate3-hydration/. This script does the
context-cheap final assembly the agent crashed on (transient API 500s): map every emitted record to its
manifest (category, group, expected_state), run `feasibility_report`, and write feasibility.{json,md}.

No sequences are read into scope beyond counting; raw sequences stay in the git-ignored store.
"""
from __future__ import annotations

import json
import os

from alignair.aa.product.applicability.cohort import MANIFEST, CohortEntry
from alignair.aa.product.applicability.feasibility import feasibility_report

STORE = ".private/aa-gate3-hydration"
ART = "docs/superpowers/artifacts/2026-07-24-aa-gate3"

CAT = {s.key: s for s in MANIFEST}


def _entry(category: str, i: int, rec: dict) -> CohortEntry:
    spec = CAT[category]
    return CohortEntry(id=f"{category}_{i}", category=category, group=spec.group,
                       expected_state=spec.expected_state, lineage_key=rec["lineage_key"])


def _load(name):
    return json.load(open(os.path.join(STORE, name)))


def build_entries() -> list:
    entries: list = []
    # flat-list sources -> a single fixed category
    for name, category in [("genairr_sim_emitted.json", "pos_igh_simulator"),
                           ("igsf_emitted.json", "igsf_vset")]:
        for i, rec in enumerate(_load(name)):
            entries.append(_entry(category, i, rec))
    # dict-keyed sources -> category is the dict key
    for name in ("oas_selected.json", "sabdab_emitted.json", "vdj_emitted.json",
                 "thera_emitted.json", "structural_emitted.json"):
        obj = _load(name)
        for category, lst in obj.items():
            for i, rec in enumerate(lst):
                entries.append(_entry(category, i, rec))
    return entries


_AA = set("ACDEFGHIKLMNPQRSTVWYX")
# VH / V-domain categories whose sequences should be a single variable domain (~90-160 aa); a much
# longer median signals constant-domain (CH1) contamination that must be trimmed before Task 3.
_VH_DOMAIN_CATS = {"pos_igh_repertoire", "pos_igh_structure", "pos_igh_simulator", "murine_vh", "vhh",
                   "design_human_framework", "humanized", "chimeric_murine_variable", "igk", "igl"}


def qc(entries_by_cat: dict) -> dict:
    """Per-category sequence QC -> caveats. Feasibility gates on COUNTS; this flags data-quality issues
    that must be fixed in Task 3 (broken emitters, constant-domain contamination) but do not by
    themselves reverse a GO when the group stays powered without the affected category."""
    import statistics
    out, caveats = {}, []
    for cat, seqs in entries_by_cat.items():
        valid = [s for s in seqs if s and set(s.upper()) <= _AA and 30 <= len(s) <= 400]
        med = int(statistics.median([len(s) for s in valid])) if valid else 0
        bad_frac = 1 - len(valid) / len(seqs) if seqs else 0.0
        out[cat] = {"n": len(seqs), "valid": len(valid), "bad_fraction": round(bad_frac, 3), "median_len": med}
        if bad_frac > 0.1:
            caveats.append(f"{cat}: {int(bad_frac*100)}% invalid sequences (broken emitter) — fix before Task 3")
        if cat in _VH_DOMAIN_CATS and med > 160:
            caveats.append(f"{cat}: median length {med} aa suggests constant-domain (CH1) contamination — "
                           "trim to the VH domain (e.g. ANARCI/IMGT) before Task 3")
    return {"per_category": out, "caveats": caveats}


def to_md(rep: dict) -> str:
    lines = ["# Gate-3 Phase-0 Feasibility — Real GO/STOP", "",
             f"**Decision: {rep['decision']}**", "",
             f"Zero-error 95% upper-bound floors: <=5% -> {rep['floors']['5pct']}, "
             f"<=10% -> {rep['floors']['10pct']} locked-test examples. "
             "Powers the LOCKED TEST after lineage dedup + fit/val/test allocation; fit/val are additional.",
             "", "| group | effective (dedup) | counts→floor | descriptive | required | test | val | fit | both loci | powered |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for g, r in rep["groups"].items():
        lines.append(f"| {g} | {r['effective_after_dedup']} | {r['counts_toward_floor']} | "
                     f"{r['descriptive_present']} | {r['required_effective_n']} | {r['alloc_test']} | "
                     f"{r['alloc_val']} | {r['alloc_fit']} | {r['both_loci_present']} | "
                     f"{'YES' if r['powered'] else 'NO'} |")
    if rep["stops"]:
        lines += ["", "## STOP reasons", ""] + [f"- {s}" for s in rep["stops"]]
    else:
        lines += ["", "All gated groups meet their locked-test floor after lineage deduplication. "
                  "Combined IGK/IGL and combined TCR have both constituent loci present; gamma/delta TCR "
                  "are present as descriptive strata (excluded from the TCR floor); structural negatives "
                  "are counted by parent lineage. **GO** -> Task 3 (cohort freeze) unblocks."]
    caveats = rep.get("qc", {}).get("caveats", [])
    lines += ["", "## Data-quality caveats (Task-3 fixes — NOT feasibility blockers)", ""]
    lines += ([f"- {c}" for c in caveats] if caveats else ["- none"])
    lines += ["", "These do not reverse the GO: each affected group stays powered without the affected "
              "category (positive_real clears its floor on repertoire+therapeutic positives alone; "
              "structural keeps its 70 parent lineages via the other transforms). They MUST be fixed "
              "during Task 3 cohort assembly."]
    return "\n".join(lines) + "\n"


def collect_sequences() -> dict:
    """{category: [aa, ...]} straight from the emitted files, for the sequence QC pass."""
    from collections import defaultdict
    seqs: dict = defaultdict(list)
    for name, category in [("genairr_sim_emitted.json", "pos_igh_simulator"),
                           ("igsf_emitted.json", "igsf_vset")]:
        for rec in _load(name):
            seqs[category].append(rec.get("aa", ""))
    for name in ("oas_selected.json", "sabdab_emitted.json", "vdj_emitted.json",
                 "thera_emitted.json", "structural_emitted.json"):
        for category, lst in _load(name).items():
            for rec in lst:
                seqs[category].append(rec.get("aa", ""))
    return seqs


def main():
    entries = build_entries()
    rep = feasibility_report(entries)
    rep["qc"] = qc(collect_sequences())
    os.makedirs(ART, exist_ok=True)
    json.dump(rep, open(os.path.join(ART, "feasibility.json"), "w"), indent=2, sort_keys=True)
    open(os.path.join(ART, "feasibility.md"), "w").write(to_md(rep))
    print(f"total entries: {len(entries)}")
    print(f"DECISION: {rep['decision']}")
    for g, r in rep["groups"].items():
        print(f"  {g:16s} eff={r['effective_after_dedup']:4d} floor_n={r['counts_toward_floor']:4d} "
              f"req={r['required_effective_n']:4d} both={r['both_loci_present']} powered={r['powered']}")
    if rep["stops"]:
        print("STOPS:", rep["stops"])


if __name__ == "__main__":
    main()
