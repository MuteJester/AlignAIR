"""Side-by-side DNAlignAIR vs IgBLAST on identical GenAIRR cases — find and characterize
exactly where we LOSE, to drive targeted architecture work.

For each stratum it scores both tools per gene (top1-in-truth-set), reports the gap, and
isolates the LOSS cases (IgBLAST correct AND DNAlignAIR wrong). It then characterizes the V
losses: SHM burden, V-segment length, and whether our wrong call is a same-gene SIBLING
(discrimination failure) or a different gene (retrieval/segmentation failure).
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records, run_igblast, igblast_to_pred, GENES  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402

STRATA = [
    ("clean",          0.0, None, None),
    ("moderate",       0.5, None, None),
    ("hard",           1.0, None, None),
    ("heavy_shm.25",   1.0, None, {"mutation_rate": 0.25}),
    ("fragment_120",   1.0, 120,  None),
    ("fragment_80",    1.0, 80,   None),
    ("extreme_trim5",  0.5, None, {"end_loss_5": (60, 120)}),
    ("heavy_shm_frag", 1.0, 80,   {"mutation_rate": 0.22}),
]


def gene_of(c):
    return c.split("*")[0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--calibration", default=".private/models/allele_set_calibration.json")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ck = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ck["config"]); m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    cal = json.load(open(args.calibration)) if os.path.exists(args.calibration) else None

    loss_rows = []   # per V-loss case (igb right, DA wrong) for characterization
    print(f"{'stratum':16s} {'gene':>4} {'DAlign':>7} {'IgBLAST':>8} {'gap':>7} {'DA-loss%':>9}")
    for name, p, crop, ov in STRATA:
        recs = gen_records(p, args.n, args.seed, crop, overrides=ov)
        reads = [r["sequence"] for r in recs]
        igb = [igblast_to_pred(x) for x in run_igblast(recs)]
        da = predict_reads(m, rs, reads, device=device, rerank="learned", calibration=cal)
        for g in GENES:
            da_ok = igb_ok = loss = tot = 0
            for rec, dp, ip in zip(recs, da, igb):
                truth = rec.get(f"{g}_call")
                if not truth:
                    continue
                tset = set(str(truth).split(","))
                dok = dp.get(f"{g}_call") in tset
                iok = (ip or {}).get(f"{g}_call") in tset
                tot += 1; da_ok += dok; igb_ok += iok
                if iok and not dok:
                    loss += 1
                    if g == "v":
                        loss_rows.append({
                            "stratum": name, "mut": float(rec.get("n_v_mutations", 0)),
                            "mutation_rate": float(rec.get("mutation_rate", 0) or 0),
                            "vlen": (rec.get("v_sequence_end", 0) or 0) - (rec.get("v_sequence_start", 0) or 0),
                            "truth_gene": gene_of(str(truth).split(",")[0]),
                            "da_call": dp.get("v_call"), "igb_call": (ip or {}).get("v_call")})
            if tot:
                da_a, ig_a = da_ok / tot, igb_ok / tot
                print(f"{name:16s} {g.upper():>4} {da_a:7.3f} {ig_a:8.3f} {da_a-ig_a:+7.3f} {loss/tot*100:8.1f}%")

    # ---- characterize the V losses ----
    print(f"\n=== V-LOSS characterization (n={len(loss_rows)} cases where IgBLAST right, DAlign wrong) ===")
    if loss_rows:
        import statistics as st
        sib = sum(1 for r in loss_rows if gene_of(r["da_call"]) == r["truth_gene"])
        print(f"  median SHM mutations in V: {st.median([r['mut'] for r in loss_rows]):.0f}")
        print(f"  median V-segment length:   {st.median([r['vlen'] for r in loss_rows]):.0f}")
        print(f"  our wrong call is a SAME-GENE sibling: {sib/len(loss_rows)*100:.0f}%  "
              f"(discrimination failure)")
        print(f"  our wrong call is a DIFFERENT gene:    {(len(loss_rows)-sib)/len(loss_rows)*100:.0f}%  "
              f"(retrieval/segmentation failure)")
        by_strat = Counter(r["stratum"] for r in loss_rows)
        print("  losses by stratum: " + ", ".join(f"{k}={v}" for k, v in by_strat.most_common()))
        by_gene = Counter(r["truth_gene"] for r in loss_rows)
        print("  top-8 most-lost V genes: " + ", ".join(f"{k}={v}" for k, v in by_gene.most_common(8)))


if __name__ == "__main__":
    main()
