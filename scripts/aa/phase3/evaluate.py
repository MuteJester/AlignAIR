"""Phase-3 evaluation runner: the frozen strategic AA model, run on the real therapeutic-antibody
VH cohort (16 lineages), scored against the frozen Phase-3 contract.

This is a READ-ONLY consumer of already-frozen artifacts (the Stage-1 model checkpoint, the Stage-2
conformal calibration, and the Phase-3 input bundle). It never re-fits, re-curates, or re-trains
anything -- it hard-fails if any of those frozen artifacts don't match the provenance recorded when
they were produced, rather than silently proceeding with a stale or substituted snapshot.

Pipeline (see docs/superpowers/plans/2026-07-21-aa-phase3-realprotein.md, Task 9):
  1. Load calibration.json; assert model sha256 == provenance.model_sha256; assert alpha matches.
  2. Assert class_maps["V"].db_hash == provenance.class_map_db_hash.V.
  2b. ASC->IMGT bridge: the model's V class map is keyed by OGRDB ASC allele names (e.g.
      "IGHVF1-G1*01"), not IMGT -- build an ImgtView translating class_members via the
      dataconfig's asc_tables["V"] provenance table (alignair.aa.taxonomy) and use ONLY that
      translated view for any family/gene scoring below.
  3. Verify bundle.manifest.json: recompute each file's sha256 (incl. the git-ignored
     hydrated_sequences.tsv) and the overall bundle hash.
  4. Load + validate the cohort (gold labels, hydrated sequences, crosswalk); assert cohort_sha256
     matches the frozen manifest.
  5. Build a FrozenConformal("V", "full") calibrator (no re-fitting).
  6. Predict V-class probabilities on the hydrated VH sequences (never on the committed labels).
  7. Score: one representative VH per lineage for the binding endpoint (family-level, never
     "supported" -- see phase3/scorer.py); concordance only if the cohort actually has
     concordance-tier rows (it does not, currently).
  8. Write a git-ignored per-row predictions.jsonl (no raw sequence) and a committed aggregate
     phase3_eval.json.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/phase3/evaluate.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

import GenAIRR as ga

from alignair.aa.conformal import FrozenConformal
from alignair.aa.infer import class_probs
from alignair.aa.phase3.cohort import (
    cohort_sha256, load_crosswalk, load_hydrated, load_labels, validate_cohort,
)
from alignair.aa.phase3.scorer import Prediction, score_concordance, score_gold
from alignair.aa.strategic import strategic_setup
from alignair.aa.taxonomy import ImgtView, asc_to_imgt_map, catalogue_families, class_families

DC_NAME = "HUMAN_IGH_OGRDB"

MODEL = "experiments/aa/strategic/model_stage1_PASS.pt"
CALIBRATION = "docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json"
ART = Path("docs/superpowers/artifacts/2026-07-21-aa-phase3")


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_family(pred: Prediction, cmap) -> set:
    """Union of IMGT families covered by a conformal prediction SET (mirrors scorer._set_family)."""
    return set().union(*[class_families(c, cmap) for c in pred.set_classes]) if pred.set_classes else set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--calibration", default=CALIBRATION)
    ap.add_argument("--art", default=str(ART))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    art = Path(args.art)

    # ---- Step 1: frozen calibration + model provenance (hard-fail on any mismatch) ----
    with open(args.calibration) as f:
        cal = json.load(f)
    model_sha = _sha256_file(args.model)
    expect_model_sha = cal["provenance"]["model_sha256"]
    assert model_sha == expect_model_sha, (
        f"PRECONDITION FAILED: model sha256 {model_sha} != calibration.json "
        f"provenance.model_sha256 {expect_model_sha}")
    assert cal["alpha"] == args.alpha, (
        f"PRECONDITION FAILED: calibration.json alpha {cal['alpha']} != expected {args.alpha}")

    model, cfg, class_maps, meta = strategic_setup(DC_NAME, 192, has_d=True)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
    model.to(args.device).eval()

    # ---- Step 2: class-map fingerprint must match the fingerprint the calibration was fit under ----
    v_db_hash = class_maps["V"].db_hash
    expect_v_db_hash = cal["provenance"]["class_map_db_hash"]["V"]
    assert v_db_hash == expect_v_db_hash, (
        f"PRECONDITION FAILED: class_maps['V'].db_hash {v_db_hash} != calibration.json "
        f"provenance.class_map_db_hash.V {expect_v_db_hash}")

    # The model's V class map is keyed by OGRDB ASC allele names (e.g. "IGHVF1-G1*01"), not IMGT --
    # gold labels are IMGT families, so translate class_members via the dataconfig's authoritative
    # ASC->IMGT provenance table before any family/gene scoring (see taxonomy.ImgtView).
    dc = getattr(ga, DC_NAME)
    vmap = ImgtView(class_maps["V"], asc_to_imgt_map(dc))

    # ---- Step 3: verify the input-bundle manifest (every listed file, incl. git-ignored hydration) ----
    with open(art / "bundle.manifest.json") as f:
        manifest = json.load(f)
    per_file = {}
    for fname, expect_hash in manifest["files"].items():
        actual_hash = _sha256_file(art / fname)
        assert actual_hash == expect_hash, (
            f"PRECONDITION FAILED: bundle file {fname!r} sha256 {actual_hash} != "
            f"bundle.manifest.json {expect_hash}")
        per_file[fname] = actual_hash
    overall = hashlib.sha256("".join(per_file[f] for f in manifest["files"]).encode()).hexdigest()
    assert overall == manifest["bundle_sha256"], (
        f"PRECONDITION FAILED: recomputed bundle_sha256 {overall} != "
        f"bundle.manifest.json bundle_sha256 {manifest['bundle_sha256']}")

    # ---- Step 4: load + validate the cohort; cohort hash must match the frozen manifest ----
    gold, audit = load_labels(str(art / "gold_labels.tsv"))
    hydrated = load_hydrated(str(art / "hydrated_sequences.tsv"))
    crosswalk = load_crosswalk(str(art / "alias_crosswalk.tsv"))
    errs = validate_cohort(gold, hydrated, crosswalk)
    assert not errs, f"PRECONDITION FAILED: cohort validation errors: {errs}"
    cohort_hash = cohort_sha256(gold, hydrated)
    assert cohort_hash == manifest["cohort_sha256"], (
        f"PRECONDITION FAILED: recomputed cohort_sha256 {cohort_hash} != "
        f"bundle.manifest.json cohort_sha256 {manifest['cohort_sha256']}")

    # ---- Step 5: frozen conformal calibrator (V gene, full-coverage Mondrian group; no re-fit) ----
    fc = FrozenConformal.from_calibration_json(cal, "V", "full")

    # ---- Step 6/7: predict on the hydrated VH (never on the committed labels) + build predictions ----
    seqs = [hydrated[r.exact_vh_id] for r in gold]
    probs = class_probs(model, seqs, cfg, args.device, "v_class_logits")
    preds = [Prediction(top_class=int(row.argmax()), set_classes=fc.predict_set(row)) for row in probs]

    # ---- Step 8: score (IMGT-translated vmap for ALL scoring, never the raw ASC-keyed class map) ----
    gold_score = score_gold(gold, preds, vmap, alpha=args.alpha, seed=0)
    decision = gold_score["binding"]["decision"]
    assert decision in ("failed", "inconclusive"), (
        f"PRECONDITION FAILED: gold binding decision {decision!r} must be 'failed' or "
        f"'inconclusive' (family endpoint must never emit 'supported')")

    concordance_rows = [r for r in gold if r.tier == "concordance"]
    if concordance_rows:
        concordance_preds = [p for r, p in zip(gold, preds) if r.tier == "concordance"]
        concordance = score_concordance(concordance_rows, concordance_preds, vmap, alpha=args.alpha)
    else:
        concordance = {"n_rows": 0, "note": "no concordance tier curated in this cohort"}

    # ---- Step 9: write outputs ----
    # git-ignored per-row predictions (no raw sequence, keyed by exact_vh_id)
    pred_lines = []
    for r, p in zip(gold, preds):
        pred_lines.append({
            "exact_vh_id": r.exact_vh_id,
            "name": r.name,
            "true_v_family": r.true_v_family,
            "top_class": p.top_class,
            "top_class_families": sorted(class_families(p.top_class, vmap)),
            "set_size": len(p.set_classes),
            "family_covered": r.true_v_family in _set_family(p, vmap),
        })
    pred_lines.sort(key=lambda d: d["exact_vh_id"])
    with open(art / "predictions.jsonl", "w") as f:
        for row in pred_lines:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    audit_reasons = Counter(a.get("_reasons", "") for a in audit)
    family_distribution = dict(Counter(r.true_v_family for r in gold))

    report = {
        "alpha": args.alpha,
        "decision": decision,
        "gold": gold_score,
        "concordance": concordance,
        "family_distribution": family_distribution,
        "asc_bridge": {
            "source": "GenAIRR HUMAN_IGH_OGRDB (OGRDB V8) asc_tables",
            "note": "model ASC allele names translated to IMGT via reference provenance",
        },
        "licensing": {
            "thera_sabdab": "redistribution not established; manifest+hydration",
            "imgt": "CC BY 4.0",
        },
        "n_audit": len(audit),
        "audit_reasons": dict(audit_reasons),
        "n_gold": len(gold),
        "bundle_sha256": manifest["bundle_sha256"],
        "cohort_sha256": manifest["cohort_sha256"],
        "provenance": {
            "model": args.model,
            "model_sha256": model_sha,
            "calibration": args.calibration,
            "class_map_v_db_hash": v_db_hash,
            "catalogue_v_families": sorted(catalogue_families(vmap)),
        },
    }
    with open(art / "phase3_eval.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    b = gold_score["binding"]
    print(f"\n==== Phase-3 real-protein eval (n_lineages={b['n_lineages']}, decision={decision}) ====")
    print(f"  top_class_family_accuracy   {b['top_class_family_accuracy']:.3f} "
          f"CI{tuple(round(x, 3) for x in b['top_class_family_ci'])}")
    print(f"  family_set_coverage         {b['family_set_coverage']:.3f} "
          f"CI{tuple(round(x, 3) for x in b['family_set_coverage_ci'])}")
    print(f"  set_size (classes) mean     {gold_score['set_size']['classes']['mean']:.2f}")
    print(f"  concordance                 {concordance}")
    print(f"\nwrote {art / 'phase3_eval.json'} (committed) and {art / 'predictions.jsonl'} (git-ignored)")


if __name__ == "__main__":
    main()
