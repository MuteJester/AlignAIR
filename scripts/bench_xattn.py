"""Wide-picture benchmark of a trained XAttnAligner checkpoint via the `alignair.benchmark` assay
suite. Generates the 22-stratum GenAIRR IGH assay online (scaled to ~millions of sequences so every
stratum/case is strongly tested), runs the model, and prints the weakness map: overall V/D/J +
coords, per-stratum worst-first, allele confusions, boundary failure decomposition, throughput.

Run (2M sequences):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src .venv/bin/python scripts/bench_xattn.py \
      --model .private/models/xattn_igh.pt --n-per-stratum 90000 --n-per-focus 90000 \
      --out .private/bench/xattn_igh_2M.json
"""
import argparse
import json
import os

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.benchmark import default_igh_assay_spec, run_online_benchmark
from alignair.benchmark.evaluation.model_adapters import xattn_predictor


def _g(d, *path, default=float("nan")):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/xattn_igh.pt")
    ap.add_argument("--n-per-stratum", type=int, default=90000)
    ap.add_argument("--n-per-focus", type=int, default=90000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--contract-level", default="core", choices=["minimal", "core", "assay"])
    ap.add_argument("--seed-m", type=int, default=0, help="k-mer seed admission (0=pure trained retrieval+matcher)")
    ap.add_argument("--cand-chunk", type=int, default=4)
    ap.add_argument("--out", default=".private/bench/xattn_igh.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = XAttnAligner(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    total = (a.n_per_stratum * 22) + a.n_per_focus
    print(f"model {a.model} (step {ck.get('step')}) | device {device} | ~{total:,} sequences "
          f"| seed_m={a.seed_m}", flush=True)

    pred = xattn_predictor(model, rs, device=device, batch_size=a.batch_size,
                           seed_m=a.seed_m, cand_chunk=a.cand_chunk)
    spec = default_igh_assay_spec(n_per_stratum=a.n_per_stratum, n_per_focus=a.n_per_focus)
    report = run_online_benchmark(spec, predictor=pred, batch_size=a.batch_size,
                                  contract_level=a.contract_level)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, default=str)
    print(f"full report -> {a.out}\n", flush=True)

    ov = report["results"]["overall"]
    perf = report.get("performance", {})
    print(f"=== OVERALL ({ov.get('n_cases')} cases) ===")
    print(f"  global: productive_acc={_g(ov,'global','productive_acc'):.3f}  "
          f"orientation_acc={_g(ov,'global','orientation_acc'):.3f}")
    for gn in ("v", "d", "j"):
        g = ov["genes"].get(gn, {})
        print(f"  {gn.upper()}: top1_in_set={_g(g,'call_top1_in_set'):.3f}  "
              f"gene_top1={_g(g,'gene_top1_in_set'):.3f}  set_f1={_g(g,'call_set_f1'):.3f}  "
              f"ss_mae={_g(g,'ss_mae'):.2f} se_mae={_g(g,'se_mae'):.2f} "
              f"gs_mae={_g(g,'gs_mae'):.2f} ge_mae={_g(g,'ge_mae'):.2f}")
    print(f"  perf: {_g(perf,'reads_per_second'):.1f} reads/s")

    print("\n=== PER-STRATUM (V/D/J top1_in_set) — worst V first ===")
    rows = []
    for ctx, m in report["results"].get("by_context", {}).items():
        if ctx.startswith("stratum:"):
            rows.append((_g(m, "genes", "v", "call_top1_in_set"), ctx.split(":", 1)[1], m))
    for gv, name, m in sorted(rows, key=lambda r: (r[0] if r[0] == r[0] else 1.0)):
        print(f"  {name:24s} V={gv:.3f}  D={_g(m,'genes','d','call_top1_in_set'):.3f}  "
              f"J={_g(m,'genes','j','call_top1_in_set'):.3f}")

    diag = report.get("diagnostics", {}).get("allele_calling", {}).get("genes", {})
    for gn in ("v", "d", "j"):
        conf = diag.get(gn, {}).get("allele_confusions", [])[:6]
        if conf:
            print(f"\n=== {gn.upper()} top allele confusions ===")
            for c in conf:
                print(f"    {str(c.get('truth_allele')):>16s} -> {str(c.get('pred_call')):16s} "
                      f"n={c.get('n')} rate={c.get('rate_among_truth_allele_cases', 0):.2f} [{c.get('error_kind')}]")

    asy = report.get("assay", {})
    print(f"\n=== ASSAY GRADE: {_g(asy,'summary','grade', default='?')} ===")


if __name__ == "__main__":
    main()
