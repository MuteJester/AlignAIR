"""Large-scale STREAMING benchmark: generate -> predict -> accumulate, no cases on disk.

Streams a big BenchmarkSpec (assay recipe = broad + focused hard strata) through a
DNAlignAIR checkpoint with a live tqdm progress bar, then prints a deep summary
(per-gene, weakest contexts, graceful degradation, assay grade) and saves the full report.

Example (watch it live in your terminal):
  PYTHONPATH=src .venv/bin/python scripts/big_online_benchmark.py --total 1000000
"""
import argparse
import json
import math
import os
import sys

import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(__file__))

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.benchmark.generation import default_igh_spec, default_igh_assay_spec  # noqa: E402
from alignair.benchmark.evaluation.online import run_online_benchmark  # noqa: E402
from alignair.benchmark.evaluation.model_adapters import dnalignair_predictor  # noqa: E402


def build_spec(recipe, total, seed):
    """Build the spec so the total case count is ~`total` across all strata."""
    builder = default_igh_assay_spec if recipe == "assay" else default_igh_spec
    n_strata = len(builder(n_per_stratum=1, seed=seed).strata)
    n_per = max(1, math.ceil(total / n_strata))
    return builder(n_per_stratum=n_per, seed=seed)


def print_summary(report):
    res = report.get("results", {})
    ov = res.get("overall", {}).get("genes", {})
    print("\n================ SUMMARY ================")
    print(f"cases scored: {res.get('overall', {}).get('n_cases', '?'):,}")
    for g in ("v", "d", "j"):
        m = ov.get(g, {})
        if m:
            print(f"  {g.upper()}: top1={m['call_top1_in_set']:.3f} gene={m['gene_top1_in_set']:.3f} "
                  f"set_rec={m['call_set_recall']:.3f} set_sz={m['pred_set_size']:.2f} "
                  f"hard_err={m.get('graceful_hard_error', float('nan')):.3f}")
    gl = res.get("overall", {}).get("global", {})
    if gl:
        print(f"  global: orient={gl.get('orientation_acc', 0):.3f} "
              f"productive={gl.get('productive_acc', 0):.3f} mut_mae={gl.get('mutation_rate_mae', 0):.3f}")
    bc = res.get("by_context", {})
    rows = []
    for ctx, cm in bc.items():
        for g in ("v", "d", "j"):
            mm = cm.get("genes", {}).get(g, {})
            if "call_top1_in_set" in mm and cm.get("n_cases", 0) >= 50:
                rows.append((mm["call_top1_in_set"], ctx, g.upper(), cm["n_cases"]))
    rows.sort()
    print("\n  --- 15 weakest (context, gene) by top1 ---")
    for t, ctx, g, n in rows[:15]:
        print(f"    {ctx:34s} {g} top1={t:.2f}  n={n}")
    print(f"\n  assay grade: {report.get('assay', {}).get('summary', {}).get('grade', '?')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--calibration", default=".private/models/allele_set_calibration.json")
    ap.add_argument("--recipe", choices=("broad", "assay"), default="assay")
    ap.add_argument("--total", type=int, default=1_000_000)
    ap.add_argument("--rerank", choices=("learned", "none"), default="learned")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="experiments/online_1M_report.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ck = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ck["config"]) if isinstance(ck["config"], dict) else ck["config"]
    model = DNAlignAIR(cfg).to(device); model.load_state_dict(ck["model"]); model.eval()
    cal = json.load(open(args.calibration)) if os.path.exists(args.calibration) else None

    spec = build_spec(args.recipe, args.total, args.seed)
    total = sum(s.n for s in spec.strata)
    print(f"streaming {total:,} cases | {len(spec.strata)} strata | recipe={args.recipe} | "
          f"rerank={args.rerank} | model={os.path.basename(args.model)} | device={device}", flush=True)

    base = dnalignair_predictor(model, rs, device=device, batch_size=args.batch,
                                rerank=args.rerank, calibration=cal)
    bar = tqdm(total=total, unit="seq", smoothing=0.05, dynamic_ncols=True)

    def predictor(reads):
        out = base(reads)
        bar.update(len(reads))
        return out

    report = run_online_benchmark(spec, predictor, reference_set=rs, batch_size=args.batch)
    bar.close()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f)
    print_summary(report)
    print(f"\nfull report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
