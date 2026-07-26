"""Source 3 hydration: VDJServer/ADC TRA + TRB — combined 5%-gate near-OOD negatives.

Draws TCR V-domains from FRESH repertoires (repertoire/subject disjoint from Gate-3) in the ADC snapshot.
Each real sequence is independently verified by **ANARCII in TCR mode** (chain A/B + locus + completeness
via the numbered V-domain span); the numbered domain is the stored sequence. Records are tagged
repository/study/donor for the study-level split allocator, denylist-filtered, counts-only census.
No guard/annotator inference runs.

STAGE-1 FINDING (2026-07-26): the ADC cache is effectively exhausted — its 58 fresh (unused) TRA
repertoires hold only 49-aa PARTIAL `sequence_alignment_aa` rows (the full-length domains belonged to
Gate-3's used reps). So this source, like OAS, needs a FRESH ADC API pull (vdjserver.org/airr/v1:
POST /repertoire for TRA/TRB human repertoires disjoint from Gate-3's studies/subjects, then
POST /rearrangement for full sequence_alignment_aa). The ANARCII-TCR verification + extraction here is the
reusable foundation; only the candidate source changes from cache to fresh API.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/hydrate_vdjserver.py
"""
from __future__ import annotations

import json
from pathlib import Path

from alignair.aa.guard.disjoint import build_gate3_global_denylist
from alignair.aa.guard.hydration import atomic_write_text, filter_disjoint, sha256_text

STORE = Path(".private/aa-applicability/fresh_hydration")
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_hydration")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"
GATE3_VDJ = ".private/aa-gate3-hydration/vdj_emitted.json"
REARR = {"tra": ".private/aa-gate3-hydration/vdj_tra_rearr_by_rep.json",
         "trb": ".private/aa-gate3-hydration/vdj_trb_rearr_by_rep.json"}
AA = set("ACDEFGHIKLMNPQRSTVWYX")


def _used_reps():
    emi = json.load(open(GATE3_VDJ))
    reps = set()
    for loc in emi:
        for r in emi[loc]:
            reps.add(r.get("repertoire_id"))
    return reps


def _candidates(category, vprefix):
    used = _used_reps()
    d = json.load(open(REARR[category]))
    out = []
    for rep_id, unit in d.items():
        if rep_id in used:
            continue
        meta = unit.get("meta", {})
        study = meta.get("study") or "unknown"
        subject = meta.get("subject_id") or "unknown"
        for row in unit.get("rows", []):
            vcall = row.get("v_call") or ""
            if not vcall.startswith(vprefix):
                continue
            aa = (row.get("sequence_alignment_aa") or "").replace(".", "").replace("-", "").replace("*", "").upper()
            if not (80 <= len(aa) <= 170) or set(aa) - AA:
                continue
            lk = f"{rep_id}::{vcall}::{row.get('cdr3_aa') or ''}"
            out.append({"lineage_key": lk, "aa": aa, "category": category, "group": "tcr",
                        "repository": "vdjserver", "study": study, "donor": subject})
    return out


def _anarcii_tcr(cands):
    if not cands:
        return []
    from anarcii import Anarcii
    model = Anarcii(seq_type="tcr", mode="speed", batch_size=64, cpu=True, ncpu=1, verbose=False)
    named = [(c["lineage_key"], c["aa"]) for c in cands]
    res = model.number(named)
    if res is None:
        return []
    got = res if isinstance(res, dict) else dict(enumerate(res))
    expect = {"tra": "A", "trb": "B"}
    by_lk = {c["lineage_key"]: c for c in cands}
    out = []
    for name, rec in (got.items() if isinstance(got, dict) else enumerate(got)):
        if not isinstance(rec, dict) or rec.get("error"):
            continue
        c = by_lk.get(name)
        if not c:
            continue
        ct, qs, qe = rec.get("chain_type"), rec.get("query_start"), rec.get("query_end")
        if qs is None or qe is None or ct != expect[c["category"]]:
            continue
        dom = c["aa"][qs:qe + 1]
        if len(dom) < 95 or set(dom) - AA:
            continue
        out.append({**c, "aa": dom, "id": f"{c['category']}_{sha256_text(c['lineage_key'])[:12]}",
                    "chain_type": ct, "verification": "anarcii_tcr_chain_locus_completeness (independent)"})
    return out


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    denylist = build_gate3_global_denylist(CLEANED, json.load(open(FP)))

    collected = []
    fresh_reps = {}
    for cat, vp in (("tra", "TRAV"), ("trb", "TRBV")):
        cands = _candidates(cat, vp)
        fresh_reps[cat] = len({c["lineage_key"].split("::")[0] for c in cands})
        collected += _anarcii_tcr(cands)

    seen, dd = set(), []
    for r in collected:
        if r["lineage_key"] in seen:
            continue
        seen.add(r["lineage_key"])
        dd.append(r)
    kept, dropped = filter_disjoint(dd, denylist=denylist, phase3=({}, {}))
    atomic_write_text(STORE / "vdjserver.normalized.json", json.dumps(kept))

    census = {
        "source": "vdjserver",
        "provenance": {"cache": REARR, "draw": "fresh repertoires (Gate-3 reps excluded)",
                       "verification": "ANARCII TCR-mode chain/locus/completeness on numbered V domain"},
        "fresh_repertoires": fresh_reps,
        "by_category": {c: sum(1 for r in kept if r["category"] == c) for c in ("tra", "trb")},
        "kept_after_denylist": len(kept),
        "denylist_dropped": dropped,
        "distinct_donors": len({(r["study"], r["donor"]) for r in kept}),
    }
    atomic_write_text(ART / "vdjserver.census.json", json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
