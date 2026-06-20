"""Fit the multi-label equivalence-set calibration (per-gene temperature + epsilon).

Runs a trained model over a LABELED GenAIRR calibration stream spanning the hard contexts
(ambiguous sets, fragments, heavy SHM), fits a per-gene temperature by multi-positive NLL,
sweeps epsilon to the smallest set achieving the target recall, and writes a sidecar
allele_set_calibration.json that predict_reads(calibration=...) consumes.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.benchmark.evaluation.allele_calibration import (  # noqa: E402
    collect_calibration_rows, fit_calibration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_novel.pt")
    ap.add_argument("--out", default=".private/models/allele_set_calibration.json")
    ap.add_argument("--n", type=int, default=250, help="records per stratum")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--objective", choices=("f1", "recall"), default="f1")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--min-recall", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ckpt = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DNAlignAIR(cfg).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # REPRESENTATIVE deployment mix (not hard-weighted) so the set isn't bloated on easy
    # data: clean/moderate/hard full reads + fragments (where the set genuinely matters) +
    # a heavy-SHM tail. Mirrors the broad benchmark distribution.
    strata = [(0.0, None, None), (0.3, None, None), (0.6, None, None), (1.0, None, None),
              (1.0, None, {"mutation_rate": 0.25}),
              (1.0, 120, None), (1.0, 80, None), (1.0, 50, None)]
    records = []
    for j, (p, crop, ov) in enumerate(strata):
        records += gen_records(p, args.n, args.seed + j, crop, overrides=ov)
    print(f"calibrating on {len(records)} labeled records ({args.model}), objective={args.objective} ...")

    rows = collect_calibration_rows(model, rs, records, topk=args.topk, device=device)
    cal = fit_calibration(rows, objective=args.objective,
                          target_recall=args.target_recall, min_recall=args.min_recall)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\nsaved -> {args.out}")
    for G, c in cal.items():
        print(f"  {G}: T={c['temperature']:.2f}  eps={c['epsilon']:.2f}  "
              f"set_size={c['mean_set_size']:.2f}  recall={c['set_recall']:.3f}  "
              f"f1={c.get('set_f1', float('nan')):.3f}  "
              f"topk_truth_recall={c['topk_truth_recall']:.3f}  n={c['n']}")


if __name__ == "__main__":
    main()
