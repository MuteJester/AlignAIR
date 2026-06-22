"""Decisive reader experiment: on the SAME k=48 candidate set, compare our learned soft-DP
reranker against an IgBLAST-style RAW per-position mismatch rescore.

If raw-mismatch rescoring closes the reader gap (0.878 -> ~0.97), the fix is cheap: inject /
upweight a raw-nucleotide-mismatch channel in the reranker (no retrain). If it does NOT,
sibling discrimination under heavy SHM genuinely needs learned changes (training).
"""
import argparse
import json
import os
import sys

import torch
from Bio import Align

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records, igblast_to_pred, run_igblast  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402


def make_aligner() -> "Align.PairwiseAligner":
    """BLAST-like local aligner (proper gapped alignment, IgBLAST-style)."""
    a = Align.PairwiseAligner()
    a.mode = "local"
    a.match_score = 2.0
    a.mismatch_score = -1.0
    a.open_gap_score = -3.0
    a.extend_gap_score = -1.0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--calibration", default=".private/models/allele_set_calibration.json")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mutation-rate", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=48)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    vref = rs.gene("V")
    germ = dict(zip(vref.names, vref.sequences))
    ck = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ck["config"]); m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    cal = json.load(open(args.calibration)) if os.path.exists(args.calibration) else None

    recs = gen_records(1.0, args.n, args.seed, None, overrides={"mutation_rate": args.mutation_rate})
    reads = [r["sequence"] for r in recs]
    igb = [igblast_to_pred(x) for x in run_igblast(recs)]
    da = predict_reads(m, rs, reads, device=device, topk=args.topk, rerank="learned", calibration=cal)
    aln = make_aligner()

    tot = learned_ok = raw_ok = oracle = igb_ok = 0
    for rec, ip, p in zip(recs, igb, da):
        truth = rec.get("v_call")
        if not truth:
            continue
        tset = set(str(truth).split(","))
        cands = p.get("v_topk", [])
        if not cands:
            continue
        tot += 1
        learned_ok += p.get("v_call") in tset
        igb_ok += (ip or {}).get("v_call") in tset
        oracle += bool(tset & set(cands))             # ceiling: truth anywhere in candidate set
        vs, ve = rec.get("v_sequence_start", 0) or 0, rec.get("v_sequence_end", 0) or 0
        seg = rec["sequence"][vs:ve].upper()
        scored = [(aln.score(seg, germ[c]), c) for c in cands if c in germ]
        if scored:
            raw_call = max(scored)[1]              # highest alignment score (IgBLAST-style)
            raw_ok += raw_call in tset

    print(f"\nRAW-ALIGNMENT rescore experiment | mr={args.mutation_rate} | k={args.topk} | n={tot}")
    print(f"  candidate-set ceiling (recall@k)             : {oracle/tot:.3f}")
    print(f"  learned soft-DP reranker (deployed)           : {learned_ok/tot:.3f}")
    print(f"  RAW gapped-local-alignment rescore (BLAST-ish): {raw_ok/tot:.3f}")
    print(f"  IgBLAST                                       : {igb_ok/tot:.3f}")
    print(f"\n  raw-align vs learned on same candidates: {float(raw_ok-learned_ok)/tot:+.3f}")


if __name__ == "__main__":
    main()
