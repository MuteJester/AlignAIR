"""Stratified benchmark of a trained AA IGH model over the real benchmark strata.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/benchmark_aa_igh.py --model experiments/aa/same_scheme/model.pt
"""
import argparse
import json
import os

import torch

from alignair_benchmark import default_igh_spec

from alignair.aa.benchmark import run_aa_benchmark
from alignair.aa.data import build_class_maps
from alignair.aa.train_gym import setup_aa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/aa/same_scheme/model.pt")
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--n-per-stratum", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/aa/same_scheme/benchmark.json")
    args = ap.parse_args()

    model, ref, cfg, _ = setup_aa(args.dc, 192)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
    model.to(args.device)
    cmaps, _ = build_class_maps(args.dc)
    eval_class_maps = {"V": cmaps["V"].allele_class, "J": cmaps["J"].allele_class}

    spec = default_igh_spec(n_per_stratum=args.n_per_stratum, seed=args.seed)
    r = run_aa_benchmark(model, ref, cfg, eval_class_maps, spec=spec, device=args.device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=2, sort_keys=True)

    print(f"\n==== AA IGH stratified benchmark ({r['spec']}) ====")
    print(f"overall AA-evaluable: {r['n_aa']}/{r['n_total']} "
          f"(no-AA {r['overall_no_aa_rate']:.0%})\n")
    hdr = f"{'stratum':22s} {'no_aa':>6} {'Vnt':>5} {'Vcls':>5} {'Vin<cls':>7} {'D':>5} {'J':>5}"
    print(hdr); print("-" * len(hdr))
    # AA-evaluable strata first, worst V-class last
    rows = [(s, d) for s, d in r["by_stratum"].items() if d["n_aa"] > 0]
    rows.sort(key=lambda kv: kv[1]["alleles"]["v"].get("class_top1", 0), reverse=True)
    for s, d in rows:
        v, dd, j = d["alleles"]["v"], d["alleles"]["d"], d["alleles"]["j"]
        print(f"{s:22s} {d['no_aa_rate']:5.0%} {v['nt_top1']:5.2f} "
              f"{v.get('class_top1', 0):5.2f} {v.get('frac_errors_within_class', 0):7.2f} "
              f"{dd['nt_top1']:5.2f} {j['nt_top1']:5.2f}  (n={d['n_aa']})")
    print("\n  not AA-evaluable (no sequence_aa or reverse-complement):")
    for s, d in r["by_stratum"].items():
        if d["n_aa"] == 0:
            print(f"    {s:22s} no_aa {d['no_aa_rate']:.0%}  non_forward {d['n_non_forward']}/{d['n_total']}")
    o = r["overall_alleles"]
    print(f"\n  OVERALL (AA-evaluable): V nt {o['v']['nt_top1']:.3f} / class {o['v']['class_top1']:.3f}"
          f"  D {o['d']['nt_top1']:.3f}  J {o['j']['nt_top1']:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
