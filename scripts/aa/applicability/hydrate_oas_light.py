"""Source 2 hydration: OAS human light chains (IGK + IGL) — combined 5%-gate near-OOD negatives.

Draws DONOR-disjoint fresh light-chain V-domains: every Gate-3 donor (normalized study::subject) is
excluded, and every record is denylist-filtered against the verified pre-split Gate-3 corpus. Loci are
kept balanced so neither IGK nor IGL can dominate the combined gate. Counts-only census. No guard model.

STAGE-1 FINDING (2026-07-26): the Gate-3 OAS-light *cache* is donor-EXHAUSTED — Gate-3 used 54 of its 57
donors, leaving only 3 disjoint (Eccles/King/Mukhamedova). Cache reuse cannot supply a donor-diverse
IGK/IGL LOCO cohort. This source must therefore be fed by a FRESH OAS pull of units from studies/donors
Gate-3 never touched (`fetch_units` below is the injection point; the extraction + donor-disjoint +
denylist logic here is the reusable foundation). The same applies to Sources 3 (VDJServer TCR) and 5
(OAS heavy).

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/hydrate_oas_light.py [--units FRESH.json]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from alignair.aa.guard.disjoint import build_gate3_global_denylist
from alignair.aa.guard.hydration import atomic_write_text, filter_disjoint, sha256_bytes, sha256_text

STORE = Path(".private/aa-applicability/fresh_hydration")
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_hydration")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"
OAS_LIGHT = ".private/aa-gate3-hydration/oas_light_raw_units.json"

PER_LOCUS_TARGET = 160   # >= 2 x 37 per subset with margin; balanced across IGK/IGL


def _norm_study(s):
    """Normalize a study label to (surname, year) so 'Bender et al., 2020' and 'Bender_2020' match."""
    s = (s or "").lower()
    surname = re.split(r"[ _,]", s.strip())[0]
    yr = re.search(r"(?:19|20)\d\d", s)
    return surname, (yr.group() if yr else "")


def _donor_key(study, subject):
    sn, yr = _norm_study(study)
    return (sn, yr, (subject or "").strip().lower())


def _used_donors():
    donors = set()
    for line in open(CLEANED):
        e = json.loads(line)
        if e.get("category") in ("igk", "igl"):
            parts = e["lineage_key"].split("::")
            if len(parts) >= 2:
                donors.add(_donor_key(parts[0], parts[1]))
    return donors


def _clean_aa(s):
    return (s or "").replace(".", "").replace("-", "").replace("*", "").upper().strip()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", default=OAS_LIGHT,
                    help="OAS units JSON (default = Gate-3 cache, which is donor-EXHAUSTED; pass a fresh pull)")
    args = ap.parse_args(argv)

    STORE.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    denylist = build_gate3_global_denylist(CLEANED, json.load(open(FP)))

    raw_bytes = Path(args.units).read_bytes()
    cache_sha = sha256_bytes(raw_bytes)
    units = json.loads(raw_bytes)
    used = _used_donors()

    picked = {"igk": {}, "igl": {}}    # lineage_key -> record (dedup within source)
    donor_skipped = 0
    for u in units:
        m = u.get("meta", {})
        study = m.get("Author") or m.get("Study") or "unknown"
        subject = m.get("Subject") or "unknown"
        if _donor_key(study, subject) in used:
            donor_skipped += 1
            continue
        for r in u.get("rows", []):
            # OAS flags are single-char strings ("T"/"F"), not booleans.
            if r.get("productive") != "T" or r.get("stop_codon") == "T" or r.get("vj_in_frame") != "T":
                continue
            vcall = (r.get("v_call") or "")
            cat = "igk" if vcall.startswith("IGKV") else "igl" if vcall.startswith("IGLV") else None
            if cat is None:
                continue
            aa = _clean_aa(r.get("sequence_alignment_aa"))
            if not (90 <= len(aa) <= 130) or set(aa) - set("ACDEFGHIKLMNPQRSTVWYX"):
                continue
            lk = f"{study}::{subject}::{vcall}::{r.get('cdr3_aa') or ''}"
            if lk in picked[cat]:
                continue
            picked[cat][lk] = {"id": f"{cat}_{sha256_text(lk)[:12]}", "aa": aa, "group": "igk_igl",
                               "category": cat, "lineage_key": lk, "donor": f"{study}::{subject}",
                               "study": study, "source": "oas_light"}

    # balanced draw: deterministic by sha256(lineage_key), cap PER_LOCUS_TARGET each
    records = []
    for cat in ("igk", "igl"):
        ordered = sorted(picked[cat].values(), key=lambda x: sha256_text(x["lineage_key"]))
        records += ordered[:PER_LOCUS_TARGET]

    kept, dropped = filter_disjoint(records, denylist=denylist, phase3=({}, {}))
    atomic_write_text(STORE / "oas_light.normalized.json", json.dumps(kept))

    census = {
        "source": "oas_light",
        "provenance": {"oas_light_cache": OAS_LIGHT, "cache_sha256": cache_sha,
                       "draw": "donor-disjoint (Gate-3 study::subject excluded) + denylist-filtered",
                       "field": "sequence_alignment_aa (IMGT gaps stripped)"},
        "gate3_donors_excluded_units": donor_skipped,
        "available_lineages": {c: len(picked[c]) for c in picked},
        "kept_after_denylist": len(kept),
        "denylist_dropped": dropped,
        "by_category": {c: sum(1 for r in kept if r["category"] == c) for c in ("igk", "igl")},
        "distinct_donors": len({r["donor"] for r in kept}),
    }
    atomic_write_text(ART / "oas_light.census.json", json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
