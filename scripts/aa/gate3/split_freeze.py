"""Task-3 FREEZE: lineage-disjoint splits, Phase-3 exclusion, RECORD-BINDING fingerprints, sealed
dev/test partitions, and audits (reqs 6-8 + review safeguards).

  6. Balanced rank-based fit/val/test splits by deterministic sha256(lineage_key). Positives .50/.25/.25;
     negatives (positive-only detector never fits on them) val/test .50/.50. Lineage-disjoint per group;
     every gated group's LOCKED-TEST slice >= floor. Category holdouts (not LOCO folds).
  7. Exclude the 16 Phase-3 antibodies from all splits (INN name + VH sha256 alias). Full accounting:
     reference / matched-and-excluded / absent / surviving.
  8. Commit only counts + provenance + fingerprints — never sequences.

Review safeguards:
  * P0 record-binding: the fingerprint hashes CANONICAL RECORDS (sequence_sha256 + verdict + source +
    transformation + split), not just lineage-key sets (see canonical.py).
  * Locked-test seal: test records go to a separate `locked_test.jsonl`; `dev_cohort.jsonl` holds only
    fit+val (see cohort_loader.py).
  * Anchor-prep circularity: motif-trimmed records are marked `prepared_by_anchor_rule`; counts exposed so
    Task 10 reports completeness FRR both including and excluding them.
  * Structural scoring: 6 transforms share a parent; score at PARENT-lineage level (a parent is a
    false-accept if ANY of its transforms is accepted) — the 35 test parents are the independent units.
  * Sequence-hash collision audit across groups/splits (no sequence carries two different expected_states).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

from alignair.aa.product.applicability import canonical
from alignair.aa.product.applicability.cohort import MANIFEST
from alignair.aa.product.applicability.feasibility import GROUP_GATE

STORE = ".private/aa-gate3-hydration"
ART = "docs/superpowers/artifacts/2026-07-24-aa-gate3"
PHASE3_GOLD = "docs/superpowers/artifacts/2026-07-21-aa-phase3/gold_labels.tsv"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _rank(keys):
    return sorted(keys, key=lambda k: _sha(k))


def _fp(keys):
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()[:16]


def _phase3():
    names, vh_by_name = set(), {}
    with open(PHASE3_GOLD) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            nm = (row.get("name") or "").strip().lower()
            if nm:
                names.add(nm)
                if row.get("exact_vh_id"):
                    vh_by_name[nm] = row["exact_vh_id"].strip()
    return names, vh_by_name


def run():
    p3_names, vh_by_name = _phase3()
    name_by_vh = {v: k for k, v in vh_by_name.items()}
    entries = [json.loads(l) for l in open(os.path.join(STORE, "cleaned_cohort.jsonl"))]

    # (7) exclusion + antibody-level accounting
    matched_names = set()
    excluded, kept = [], []
    for e in entries:
        nm = e["lineage_key"].strip().lower()
        vh = _sha(e["aa"])
        hit_name = nm in p3_names
        hit_vh = vh in name_by_vh
        if hit_name:
            matched_names.add(nm)
        if hit_vh:
            matched_names.add(name_by_vh[vh])
        if hit_name or hit_vh:
            excluded.append({"id": e["id"], "why": "phase3_inn" if hit_name else "phase3_vh_hash"})
        else:
            kept.append(e)
    assert not any(e["lineage_key"].strip().lower() in p3_names or _sha(e["aa"]) in name_by_vh
                   for e in kept), "Phase-3 leak survived exclusion"
    accounting = {"reference": len(p3_names), "matched_and_excluded": len(matched_names),
                  "absent": len(p3_names) - len(matched_names), "surviving": 0,
                  "excluded_records": len(excluded), "matched_names": sorted(matched_names)}

    # Independent unit = DISTINCT SEQUENCE (identical sequences are correlated, not independent
    # observations for the FAR/FRR bound — e.g. germline TCR V-domains recur across subjects). Dedup by
    # (group, sequence_sha256); structural keeps every parent (each parent's reversed_vh is unique, so a
    # parent is never removed). This subsumes the collision audit: after it, the only permitted repeats
    # are cross-group and must never carry conflicting verdicts (asserted below).
    seen, seq_deduped, seq_dropped = set(), [], 0
    for e in kept:
        k = (e["group"], _sha(e["aa"]))
        if k in seen:
            seq_dropped += 1
            continue
        seen.add(k)
        seq_deduped.append(e)
    kept = seq_deduped

    # (6) balanced rank-based split; structural parents restricted to val/test (never membership fit)
    role_of = {g: r for g, (_, r) in GROUP_GATE.items()}
    struct_lineages = {e["lineage_key"] for e in kept if e["group"] == "structural"}
    group_lineages: dict = {}
    for e in kept:
        group_lineages.setdefault(e["group"], set()).add(e["lineage_key"])

    lk_split: dict = {}
    for g in ("positive_real", "positive_sim"):
        keys = group_lineages.get(g, set())
        fit_elig = _rank([k for k in keys if k not in struct_lineages])
        restricted = _rank([k for k in keys if k in struct_lineages])
        f, v = round(len(fit_elig) * 0.5), round(len(fit_elig) * 0.75)
        for i, k in enumerate(fit_elig):
            lk_split[(g, k)] = "fit" if i < f else "val" if i < v else "test"
        vv = round(len(restricted) * 0.5)
        for i, k in enumerate(restricted):
            lk_split[(g, k)] = "val" if i < vv else "test"
    posreal = {k: sp for (g, k), sp in lk_split.items() if g == "positive_real"}
    struct_ordered = _rank(group_lineages.get("structural", set()))
    sv = round(len(struct_ordered) * 0.5)
    for i, k in enumerate(struct_ordered):
        lk_split[("structural", k)] = posreal.get(k, "val" if i < sv else "test")
    for g, keys in group_lineages.items():
        if g in ("positive_real", "positive_sim", "structural"):
            continue
        ordered = _rank(keys)
        vv = round(len(ordered) * 0.5)
        for i, k in enumerate(ordered):
            lk_split[(g, k)] = "val" if i < vv else "test"

    for e in kept:
        e["split"] = lk_split[(e["group"], e["lineage_key"])]

    # invariants: within-group lineage disjointness + no structural parent in membership fit
    buckets: dict = {}
    for e in kept:
        buckets.setdefault((e["split"], e["group"]), set()).add(e["lineage_key"])
    for g in {grp for _, grp in buckets}:
        s = {sp: buckets.get((sp, g), set()) for sp in ("fit", "val", "test")}
        assert not (s["fit"] & s["val"]) and not (s["fit"] & s["test"]) and not (s["val"] & s["test"]), g
    assert not (buckets.get(("fit", "positive_real"), set()) & struct_lineages), "structural parent in fit"

    # sequence-hash collision audit. After per-group sequence dedup, any residual repeat is CROSS-GROUP;
    # none may carry conflicting verdicts, and none may be intra-group (that would be a dedup failure).
    by_seq: dict = {}
    for e in kept:
        by_seq.setdefault(_sha(e["aa"]), []).append(e)
    residual = {h: recs for h, recs in by_seq.items() if len(recs) > 1}
    contradictions = [h for h, recs in residual.items() if len({r["expected_state"] for r in recs}) > 1]
    assert not contradictions, f"sequence used with conflicting verdicts: {contradictions[:5]}"
    intra_group = [h for h, recs in residual.items() if len({r["group"] for r in recs}) == 1]
    assert not intra_group, f"intra-group sequence duplicate survived dedup: {intra_group[:5]}"
    collision_audit = {"removed_intra_group_duplicates": seq_dropped,
                       "residual_cross_group": len(residual), "contradictions": 0}

    # floors + anchor-prepared counts + structural parent count
    floor_ok = {g: {"test_lineages": len(buckets.get(("test", g), set())), "floor": floor,
                    "meets_floor": len(buckets.get(("test", g), set())) >= floor}
                for g, (floor, _) in GROUP_GATE.items()}
    anchor_prepared = {sp: sum(1 for e in kept if e["split"] == sp
                               and canonical.prepared_by_anchor_rule(e["category"]))
                       for sp in ("fit", "val", "test")}

    # (P0) record-binding canonical digests, per split + global
    canon = {"fit": [], "val": [], "test": []}
    for e in kept:
        canon[e["split"]].append(canonical.canonical_record(e, e["split"]))
    record_digest = {sp: canonical.digest(canon[sp]) for sp in canon}
    record_digest["global"] = canonical.digest([c for cs in canon.values() for c in cs])

    fingerprint = {
        "phase3": {**accounting,
                   "by_reason": {r: sum(1 for x in excluded if x["why"] == r)
                                 for r in ("phase3_inn", "phase3_vh_hash")}},
        "split_policy": {"pos": "fit .50 / val .25 / test .25", "neg": "val .50 / test .50",
                         "assignment": "balanced rank by sha256(lineage_key)"},
        "record_digest": record_digest,
        "sequence_collision_audit": collision_audit,
        "anchor_prepared_positives_note": "categories prepared by motif_fr4_junction_v1 are marked "
            "prepared_by_anchor_rule; Task 10 must report completeness FRR both including and excluding "
            "them, must NOT tune Stage-1 motifs/thresholds on them, and treats their result as descriptive. "
            "Primary real-positive gate = repertoire + therapeutic VH (direct-prepared).",
        "anchor_prepared_counts": anchor_prepared,
        "structural_scoring_note": "score structural at PARENT-lineage level: a parent is falsely accepted "
            "if ANY of its 6 expected-incomplete transforms is accepted. The "
            f"{floor_ok['structural']['test_lineages']} test parents are the independent units for the "
            "confidence bound — do NOT count 6 correlated transforms as 6 observations.",
        "groups": {}, "category_holdouts": {},
    }
    for g in sorted({grp for _, grp in buckets}):
        fingerprint["groups"][g] = {sp: {"n": len(buckets.get((sp, g), set())),
                                         "lineage_set_sha16": _fp(buckets.get((sp, g), set()))}
                                    for sp in ("fit", "val", "test") if (sp, g) in buckets}
        fingerprint["groups"][g]["locked_test_floor_ok"] = floor_ok[g]
    cat_test: dict = {}
    for e in kept:
        if e["split"] == "test":
            cat_test.setdefault(e["category"], set()).add(e["lineage_key"])
    for cat, keys in sorted(cat_test.items()):
        fingerprint["category_holdouts"][cat] = {"test_n": len(keys), "test_sha16": _fp(keys)}

    os.makedirs(ART, exist_ok=True)
    json.dump(fingerprint, open(os.path.join(ART, "split_fingerprint.json"), "w"), indent=2, sort_keys=True)
    # sealed partitions: dev (fit+val) vs locked test — separate files (see cohort_loader.py)
    with open(os.path.join(STORE, "dev_cohort.jsonl"), "w") as fh:
        for e in kept:
            if e["split"] in ("fit", "val"):
                fh.write(json.dumps(e) + "\n")
    with open(os.path.join(STORE, "locked_test.jsonl"), "w") as fh:
        for e in kept:
            if e["split"] == "test":
                fh.write(json.dumps(e) + "\n")
    # remove any stale unlock audit from a prior freeze so the test starts sealed
    ap = os.path.join(STORE, "locked_test_unlock_audit.jsonl")
    if os.path.exists(ap):
        os.remove(ap)

    print(f"kept {len(kept)} (seq-dedup removed {seq_dropped}) | Phase-3 reference=16 "
          f"matched&excluded={accounting['matched_and_excluded']} {accounting['matched_names']} "
          f"absent={accounting['absent']} surviving=0 (records={len(excluded)})")
    print(f"global record digest: {record_digest['global'][:16]}... | "
          f"collisions: intra-group removed {seq_dropped}, residual cross-group {len(residual)}")
    for g, d in sorted(fingerprint["groups"].items()):
        fo = d["locked_test_floor_ok"]
        print(f"  {g:16s} fit={d.get('fit', {}).get('n', 0):4d} val={d.get('val', {}).get('n', 0):4d} "
              f"test={d.get('test', {}).get('n', 0):4d}  test>=floor({fo['floor']}): {fo['meets_floor']}")
    print(f"anchor_prepared(test)={anchor_prepared['test']} | structural test parents="
          f"{floor_ok['structural']['test_lineages']} (parent-level scored)")
    assert all(f["meets_floor"] for f in floor_ok.values()), "a gated group's test slice is below floor"
    print("FREEZE OK.")
    return fingerprint


if __name__ == "__main__":
    run()
