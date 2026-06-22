"""Decide the heavy-SHM V weakness: retrieval-bound vs reader-bound.

For the heavy-SHM stratum, score against IgBLAST and split every DNAlignAIR V-loss case
(IgBLAST right, we wrong) into:
  - RETRIEVAL-bound: the true allele isn't in our top-k candidate set at all.
  - READER-bound:    the true allele IS in top-k, but our learned reranker put a wrong
                     sibling on top.
Also reports recall@k of the true allele at k=16 vs k=48, to see if simply widening
retrieval closes the gap.
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records, run_igblast, igblast_to_pred  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402


def gene_of(c):
    return c.split("*")[0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--calibration", default=".private/models/allele_set_calibration.json")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mutation-rate", type=float, default=0.25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ck = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ck["config"]); m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    cal = json.load(open(args.calibration)) if os.path.exists(args.calibration) else None

    recs = gen_records(1.0, args.n, args.seed, None, overrides={"mutation_rate": args.mutation_rate})
    reads = [r["sequence"] for r in recs]
    igb = [igblast_to_pred(x) for x in run_igblast(recs)]
    da16 = predict_reads(m, rs, reads, device=device, topk=16, rerank="learned", calibration=cal)
    da48 = predict_reads(m, rs, reads, device=device, topk=48, rerank="learned", calibration=cal)

    rec16 = rec48 = tot = 0
    acc16 = acc48 = igb_acc = 0
    retr_bound = reader_bound = 0
    reader_fams = Counter()
    for rec, ip, p16, p48 in zip(recs, igb, da16, da48):
        truth = rec.get("v_call")
        if not truth:
            continue
        tset = set(str(truth).split(","))
        tot += 1
        in16 = bool(tset & set(p16.get("v_topk", [])))
        in48 = bool(tset & set(p48.get("v_topk", [])))
        rec16 += in16; rec48 += in48
        acc16 += p16.get("v_call") in tset
        acc48 += p48.get("v_call") in tset
        igb_acc += (ip or {}).get("v_call") in tset
        # loss vs IgBLAST: IgBLAST right, we (k=16, deployed) wrong
        iok = (ip or {}).get("v_call") in tset
        dok = p16.get("v_call") in tset
        if iok and not dok:
            if not in16:
                retr_bound += 1
            else:
                reader_bound += 1
                reader_fams[gene_of(str(truth).split(",")[0])] += 1

    print(f"\nheavy-SHM V diagnostic | mutation_rate={args.mutation_rate} | n={tot}")
    print(f"  true-allele recall@16 : {rec16/tot:.3f}")
    print(f"  true-allele recall@48 : {rec48/tot:.3f}   (Δ={float(rec48-rec16)/tot:+.3f})")
    print(f"  END-TO-END V acc @16  : {acc16/tot:.3f}  (deployed)")
    print(f"  END-TO-END V acc @48  : {acc48/tot:.3f}  (Δ={float(acc48-acc16)/tot:+.3f})")
    print(f"  IgBLAST V acc         : {igb_acc/tot:.3f}")
    losses = retr_bound + reader_bound
    print(f"\n  V-losses vs IgBLAST: {losses}")
    if losses:
        print(f"    RETRIEVAL-bound (truth not in top-16): {retr_bound:3d}  ({retr_bound/losses*100:.0f}%)")
        print(f"    READER-bound    (truth in top-16, misranked): {reader_bound:3d}  ({reader_bound/losses*100:.0f}%)")
        print("    reader-bound by family: " + ", ".join(f"{k}={v}" for k, v in reader_fams.most_common(8)))


if __name__ == "__main__":
    main()
