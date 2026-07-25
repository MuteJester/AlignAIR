"""Task-3 FREEZE: lineage-disjoint fit/val/test splits with category/species holdouts, Phase-3
exclusion, and committable fingerprints (reqs 6-8).

  6. Freeze lineage / species / category-held-out splits. For the positive-only detector these are
     CATEGORY HOLDOUTS (each negative category has a test presence) — NOT LOCO retraining folds.
  7. Assert the 16 Phase-3 antibodies occur in NO split, directly (INN name) or via alias (VH hash).
  8. Commit only counts + provenance + split fingerprints. Lineage keys embed CDR3 substrings, so the
     fingerprint stores a sha256 over the sorted lineage-key set per split/group — never the keys.

Splits are BALANCED rank-based (sort lineages by a deterministic hash, slice by fraction) so small groups
still meet their locked-test floor. Positives: fit/val/test = .50/.25/.25; negatives (the positive-only
detector never fits on them): val/test = .50/.50.

Leakage handling — structural parents overlap 70/70 with repertoire positives (the fragments are derived
from those VHs). Since structural negatives are Stage-1 COMPLETENESS-rule targets (never scored by the
positive-only membership detector), the only real risk is a parent sitting in the membership FIT set while
its fragment is a test negative. We remove it structurally: shared parent lineages are restricted to
val/test (never fit), and each structural fragment is assigned the SAME split as its parent's positive
form. So no membership-fit lineage ever has a structural fragment in val/test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

from alignair.aa.product.applicability.feasibility import GROUP_GATE
from alignair.aa.product.applicability.cohort import MANIFEST

STORE = ".private/aa-gate3-hydration"
ART = "docs/superpowers/artifacts/2026-07-24-aa-gate3"
PHASE3_GOLD = "docs/superpowers/artifacts/2026-07-21-aa-phase3/gold_labels.tsv"


def _rank(keys):
    return sorted(keys, key=lambda k: hashlib.sha256(k.encode()).hexdigest())


def _fp(keys):
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()[:16]


def _phase3_identities():
    names, vh_ids = set(), set()
    with open(PHASE3_GOLD) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("name"):
                names.add(row["name"].strip().lower())
            if row.get("exact_vh_id"):
                vh_ids.add(row["exact_vh_id"].strip())
    return names, vh_ids


def run():
    p3_names, p3_vh = _phase3_identities()
    entries = [json.loads(l) for l in open(os.path.join(STORE, "cleaned_cohort.jsonl"))]

    # (7) exclude the 16 Phase-3 antibodies by INN name OR VH sha256 (alias catch)
    def _is_p3(e):
        return (e["lineage_key"].strip().lower() in p3_names
                or hashlib.sha256(e["aa"].encode()).hexdigest() in p3_vh)
    excluded = [{"id": e["id"], "category": e["category"],
                 "why": "phase3_inn" if e["lineage_key"].strip().lower() in p3_names else "phase3_vh_hash"}
                for e in entries if _is_p3(e)]
    kept = [e for e in entries if not _is_p3(e)]
    assert not any(_is_p3(e) for e in kept), "Phase-3 leak survived exclusion"

    role_of = {g: r for g, (_, r) in GROUP_GATE.items()}
    struct_lineages = {e["lineage_key"] for e in kept if e["group"] == "structural"}
    group_lineages: dict = {}
    for e in kept:
        group_lineages.setdefault(e["group"], set()).add(e["lineage_key"])

    lk_split: dict = {}                         # (group, lineage_key) -> split
    # positives: structural-parent lineages restricted to val/test (never fit); others fit/val/test
    for g in ("positive_real", "positive_sim"):
        keys = group_lineages.get(g, set())
        fit_elig = _rank([k for k in keys if k not in struct_lineages])
        restricted = _rank([k for k in keys if k in struct_lineages])
        nf = len(fit_elig)
        f, v = round(nf * 0.5), round(nf * 0.75)
        for i, k in enumerate(fit_elig):
            lk_split[(g, k)] = "fit" if i < f else "val" if i < v else "test"
        nr = len(restricted)
        vv = round(nr * 0.5)
        for i, k in enumerate(restricted):
            lk_split[(g, k)] = "val" if i < vv else "test"
    # structural: reuse the parent's positive_real split (consistency); fall back to balanced val/test
    posreal = {k: sp for (g, k), sp in lk_split.items() if g == "positive_real"}
    struct_ordered = _rank(group_lineages.get("structural", set()))
    sv = round(len(struct_ordered) * 0.5)
    for i, k in enumerate(struct_ordered):
        lk_split[("structural", k)] = posreal.get(k, "val" if i < sv else "test")
    # other negative groups: balanced val/test
    for g, keys in group_lineages.items():
        if g in ("positive_real", "positive_sim", "structural"):
            continue
        ordered = _rank(keys)
        vv = round(len(ordered) * 0.5)
        for i, k in enumerate(ordered):
            lk_split[(g, k)] = "val" if i < vv else "test"

    buckets: dict = {}                          # (split, group) -> set(lineage_key)
    cat_test: dict = {}                         # category -> set(lineage_key) in test
    for e in kept:
        sp = lk_split[(e["group"], e["lineage_key"])]
        buckets.setdefault((sp, e["group"]), set()).add(e["lineage_key"])
        if sp == "test":
            cat_test.setdefault(e["category"], set()).add(e["lineage_key"])

    # (6) lineage disjointness within each group
    for g in {grp for _, grp in buckets}:
        s = {sp: buckets.get((sp, g), set()) for sp in ("fit", "val", "test")}
        assert not (s["fit"] & s["val"]) and not (s["fit"] & s["test"]) and not (s["val"] & s["test"]), \
            f"lineage overlap in {g}"
    # no membership-fit lineage has a structural fragment (leak guard)
    fit_pos = buckets.get(("fit", "positive_real"), set())
    assert not (fit_pos & struct_lineages), "structural parent leaked into the membership fit set"

    # locked-test floor per gated group (test slice >= floor)
    floor_ok = {g: {"test_lineages": len(buckets.get(("test", g), set())), "floor": floor,
                    "meets_floor": len(buckets.get(("test", g), set())) >= floor}
                for g, (floor, _) in GROUP_GATE.items()}

    fingerprint = {
        "phase3_excluded": {"n": len(excluded),
                            "by_reason": {r: sum(1 for x in excluded if x["why"] == r)
                                          for r in ("phase3_inn", "phase3_vh_hash")}},
        "split_policy": {"pos": "fit .50 / val .25 / test .25", "neg": "val .50 / test .50",
                         "assignment": "balanced rank by sha256(lineage_key)",
                         "leakage": "structural parents (70/70 shared with repertoire positives) "
                                    "restricted to val/test; structural fragment split = parent split; "
                                    "structural negatives are Stage-1 completeness targets"},
        "groups": {}, "category_holdouts": {},
    }
    for g in sorted({grp for _, grp in buckets}):
        fingerprint["groups"][g] = {sp: {"n": len(buckets.get((sp, g), set())),
                                         "lineage_set_sha16": _fp(buckets.get((sp, g), set()))}
                                    for sp in ("fit", "val", "test") if (sp, g) in buckets}
        fingerprint["groups"][g]["locked_test_floor_ok"] = floor_ok[g]
    for cat, keys in sorted(cat_test.items()):
        fingerprint["category_holdouts"][cat] = {"test_n": len(keys), "test_sha16": _fp(keys)}

    os.makedirs(ART, exist_ok=True)
    json.dump(fingerprint, open(os.path.join(ART, "split_fingerprint.json"), "w"), indent=2, sort_keys=True)
    with open(os.path.join(STORE, "splits.jsonl"), "w") as fh:
        for e in kept:
            fh.write(json.dumps({"id": e["id"], "category": e["category"], "group": e["group"],
                                 "split": lk_split[(e["group"], e["lineage_key"])]}) + "\n")

    print(f"kept {len(kept)} | Phase-3 excluded {len(excluded)} "
          f"(inn={fingerprint['phase3_excluded']['by_reason']['phase3_inn']}, "
          f"vh={fingerprint['phase3_excluded']['by_reason']['phase3_vh_hash']})")
    for g, d in sorted(fingerprint["groups"].items()):
        fo = d["locked_test_floor_ok"]
        print(f"  {g:16s} fit={d.get('fit', {}).get('n', 0):4d} val={d.get('val', {}).get('n', 0):4d} "
              f"test={d.get('test', {}).get('n', 0):4d}  test>=floor({fo['floor']}): {fo['meets_floor']}")
    assert all(f["meets_floor"] for f in floor_ok.values()), "a gated group's test slice is below floor"
    print("FREEZE OK: lineage-disjoint, no fit-set leak, Phase-3 excluded, every gated test slice >= floor.")
    return fingerprint


if __name__ == "__main__":
    run()
