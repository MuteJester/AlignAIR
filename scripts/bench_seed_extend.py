"""Wide-picture benchmark of a trained seed_extend (or any DNAlignAIR) checkpoint via the
`alignair.benchmark` assay platform. Generates the 22-stratum GenAIRR IGH assay online, runs the
model, and prints a weakness map: overall V/D/J + coords, per-stratum worst-first, allele
confusions, boundary failure decomposition, and throughput. Full JSON is written to --out.

Run:
  PYTHONPATH=src .venv/bin/python scripts/bench_seed_extend.py \
      --model .private/models/seed_extend_d64_reader.pt --n-per-stratum 150
"""
import argparse
import json

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.reference.reference_set import ReferenceSet
from alignair.benchmark import default_igh_assay_spec, run_online_benchmark
from alignair.benchmark.evaluation.model_adapters import dnalignair_predictor


def _g(d, *path, default=float("nan")):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/seed_extend_d64_reader.pt")
    ap.add_argument("--n-per-stratum", type=int, default=150)
    ap.add_argument("--n-per-focus", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--contract-level", default="core", choices=["minimal", "core", "assay"])
    ap.add_argument("--rerank", default="learned", choices=["none", "learned"],
                    help="'none'=pooled retrieval argmax; 'learned'=in-model DP reader (seed_extend wired)")
    ap.add_argument("--v-reader", default="learned", choices=["learned", "parasail"],
                    help="reader for V: 'learned' DP, or classical parasail SW")
    ap.add_argument("--out", default=".private/bench/seed_extend_assay.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    print(f"model: {a.model} | seed_extend={getattr(model, 'seed_extend', None)} caller={model.caller} device={device}")

    pred = dnalignair_predictor(model, rs, device=device, batch_size=a.batch_size,
                                rerank=a.rerank, v_reader=a.v_reader)
    print(f"reader: rerank={a.rerank} v_reader={a.v_reader}")
    spec = default_igh_assay_spec(n_per_stratum=a.n_per_stratum, n_per_focus=a.n_per_focus)
    report = run_online_benchmark(spec, predictor=pred, batch_size=a.batch_size,
                                  contract_level=a.contract_level)

    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, default=str)
    print(f"full report -> {a.out}\n")

    ov = report["results"]["overall"]
    n_cases = ov.get("n_cases")
    perf = report.get("performance", {})
    print(f"=== OVERALL ({n_cases} cases) ===")
    print(f"  global: productive_acc={_g(ov,'global','productive_acc'):.3f}  "
          f"orientation_acc={_g(ov,'global','orientation_acc'):.3f}")
    for gn in ("v", "d", "j"):
        g = ov["genes"].get(gn, {})
        print(f"  {gn.upper()}: top1_in_set={_g(g,'call_top1_in_set'):.3f}  "
              f"gene_top1={_g(g,'gene_top1_in_set'):.3f}  set_f1={_g(g,'call_set_f1'):.3f}  "
              f"ss_mae={_g(g,'ss_mae'):.2f} se_mae={_g(g,'se_mae'):.2f} "
              f"gs_mae={_g(g,'gs_mae'):.2f} ge_mae={_g(g,'ge_mae'):.2f}")
    print(f"  perf: {_g(perf,'reads_per_second'):.1f} reads/s  "
          f"{_g(perf,'seconds_per_read')*1000:.2f} ms/read  peak_mem={_g(perf,'peak_memory_mb'):.0f}MB")

    # per-stratum, worst V top1 first
    print("\n=== PER-STRATUM (V/D/J top1_in_set, V coord MAE) — worst V first ===")
    rows = []
    for ctx, m in report["results"].get("by_context", {}).items():
        if not ctx.startswith("stratum:"):
            continue
        gv = _g(m, "genes", "v", "call_top1_in_set")
        rows.append((gv, ctx.split(":", 1)[1], m))
    for gv, name, m in sorted(rows, key=lambda r: (r[0] if r[0] == r[0] else 1.0)):
        print(f"  {name:24s} V={gv:.3f}  D={_g(m,'genes','d','call_top1_in_set'):.3f}  "
              f"J={_g(m,'genes','j','call_top1_in_set'):.3f}  "
              f"Vss_mae={_g(m,'genes','v','ss_mae'):.1f} Vge_mae={_g(m,'genes','v','ge_mae'):.1f}")

    # allele diagnostics: summary + top confusions
    diag = report.get("diagnostics", {}).get("allele_calling", {}).get("genes", {})
    for gn in ("v", "d", "j"):
        g = diag.get(gn, {})
        s = g.get("summary", {})
        print(f"\n=== {gn.upper()} ALLELE DIAGNOSTICS ===")
        print(f"  allele_acc={_g(s,'allele_accuracy'):.3f} gene_acc={_g(s,'gene_accuracy'):.3f} "
              f"family_acc={_g(s,'family_accuracy'):.3f}")
        conf = g.get("allele_confusions", [])[:8]
        for c in conf:
            print(f"    {c.get('truth_allele'):>16s} -> {str(c.get('pred_call')):16s} "
                  f"n={c.get('n')} rate={c.get('rate_among_truth_allele_cases', 0):.2f} "
                  f"[{c.get('error_kind')}]")

    # boundary failure types
    bnd = report.get("diagnostics", {}).get("boundaries", {}).get("genes", {})
    for gn in ("v", "d", "j"):
        g = bnd.get(gn, {})
        s = g.get("summary", {})
        ft = g.get("failure_types", [])[:6]
        if ft:
            print(f"\n=== {gn.upper()} BOUNDARY FAILURES "
                  f"(qss_mae={_g(s,'sequence_start_mae'):.2f} germ_start_mae={_g(s,'germline_start_mae'):.2f} "
                  f"exact_germ_span={_g(s,'exact_germline_span_rate'):.3f}) ===")
            for f in ft:
                print(f"    {str(f.get('failure_type')):36s} n={f.get('n')} rate={f.get('rate') or 0:.3f}")

    asy = report.get("assay", {})
    print(f"\n=== ASSAY GRADE: {_g(asy,'summary','grade', default='?')} ===")
    for cf in asy.get("critical_failures", [])[:10]:
        print(f"  CRITICAL: {cf}")
    for wc in asy.get("weak_contexts", [])[:10]:
        print(f"  weak: {wc}")


if __name__ == "__main__":
    main()
