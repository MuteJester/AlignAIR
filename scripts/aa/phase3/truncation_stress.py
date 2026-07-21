"""Task 10: SYNTHETIC truncation stress test on the real Gold VH cohort.

This is NOT the primary Phase-3 result (see evaluate.py / phase3_eval.json). It is a diagnostic
probe: deterministically crop each real Gold VH sequence to each frozen Mondrian coverage bin's
keep-fraction range -- full=uncropped, trunc=keep in [0.6,0.9], short=keep in [0.3,0.6), exactly the
bin definitions the Stage-2 calibration.json was fit under (see its `provenance.bin_defs`) -- score
V-class probabilities on the cropped sequence, and check family-set coverage under that BIN'S OWN
frozen conformal quantile. Cropping never changes a row's true_v_family.

The frozen `q_mondrian["trunc"]` / `["short"]` quantiles were calibrated on SIMULATOR truncations
(GenAIRR), not on truncated real antibody sequences. This script asks whether that calibration still
holds under real partial coverage -- a domain-shift stress test, reported separately from the primary
result, never merged into it.

Read-only on all frozen inputs (model checkpoint, calibration.json, gold_labels.tsv, and the
git-ignored hydrated_sequences.tsv). Writes only an aggregate per-bin coverage JSON -- no raw
sequences, no per-row predictions.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/phase3/truncation_stress.py --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import GenAIRR as ga

from alignair.aa.augment import aa_truncate
from alignair.aa.conformal import FrozenConformal
from alignair.aa.infer import class_probs
from alignair.aa.phase3.cohort import load_hydrated, load_labels
from alignair.aa.strategic import strategic_setup
from alignair.aa.taxonomy import ImgtView, asc_to_imgt_map, class_families

DC_NAME = "HUMAN_IGH_OGRDB"
MODEL = "experiments/aa/strategic/model_stage1_PASS.pt"
CALIBRATION = "docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json"
BUNDLE = Path("docs/superpowers/artifacts/2026-07-21-aa-phase3")
OUT = BUNDLE / "truncation_stress.json"

# Exact bin definitions from calibration.json's provenance.bin_defs -- do NOT invent.
# full=uncropped; trunc=keep fraction in [0.6, 0.9]; short=keep fraction in [0.3, 0.6).
BINS = {
    "full": None,
    "trunc": (0.6, 0.9),
    "short": (0.3, 0.6),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--calibration", default=CALIBRATION)
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    bundle = Path(args.bundle)

    with open(args.calibration) as f:
        cal = json.load(f)

    model, cfg, class_maps, _meta = strategic_setup(DC_NAME, 192, has_d=True)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
    model.to(args.device).eval()

    # The model's V class map is keyed by OGRDB ASC allele names, not IMGT -- translate via the
    # dataconfig's provenance table before any family scoring (exactly as evaluate.py does).
    dc = getattr(ga, DC_NAME)
    vmap = ImgtView(class_maps["V"], asc_to_imgt_map(dc))

    gold, _audit = load_labels(str(bundle / "gold_labels.tsv"))
    hydrated = load_hydrated(str(bundle / "hydrated_sequences.tsv"))

    # One rng, seeded once with the frozen seed, threaded across bins in a fixed order (full has no
    # randomness) -- a paired design: every bin scores the SAME set of Gold VH rows, in the same
    # order, so per-bin coverage is directly comparable.
    rng = np.random.default_rng(args.seed)

    by_bin = {}
    for bin_name, keep_range in BINS.items():
        if keep_range is None:
            seqs = [hydrated[r.exact_vh_id] for r in gold]
        else:
            lo, hi = keep_range
            seqs = [aa_truncate(hydrated[r.exact_vh_id], lo, hi, rng) for r in gold]

        qhat = cal["genes"]["V"]["calibration_frozen"]["q_mondrian"][bin_name]
        fc = FrozenConformal(qhat=qhat)
        probs = class_probs(model, seqs, cfg, args.device, "v_class_logits")

        covered = []
        for r, probrow in zip(gold, probs):
            predict_set = fc.predict_set(probrow)
            fam_set = set().union(*[class_families(c, vmap) for c in predict_set]) if predict_set else set()
            covered.append(r.true_v_family in fam_set)

        entry = {
            "n": len(covered),
            "family_coverage": float(np.mean(covered)),
            "quantile": qhat,
        }
        if keep_range is not None:
            entry["keep_range"] = list(keep_range)
        by_bin[bin_name] = entry

    assert set(by_bin) == {"full", "trunc", "short"}, f"unexpected bins: {sorted(by_bin)}"

    report = {
        "kind": "SYNTHETIC truncation stress test on real VH — separate from the primary result",
        "seed": args.seed,
        "by_bin": by_bin,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"\n==== Truncation stress test (SYNTHETIC, n={by_bin['full']['n']}) ====")
    for bin_name in ("full", "trunc", "short"):
        e = by_bin[bin_name]
        print(f"  {bin_name:6s}  family_coverage={e['family_coverage']:.3f}  "
              f"quantile={e['quantile']:.4f}  n={e['n']}")
    print(f"\nwrote {out_path} (committed)")


if __name__ == "__main__":
    main()
