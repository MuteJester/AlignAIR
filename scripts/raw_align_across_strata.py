"""Validate the raw-alignment V reader ACROSS strata: it must close the heavy-SHM gap
WITHOUT regressing the cases neural retrieval currently wins (fragments).

Per stratum, on the SAME k=48 candidate set, compares: learned soft-DP reader (deployed),
raw gapped-local-alignment rescore, a simple BLEND (learned z + raw z), and IgBLAST.
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

STRATA = [
    ("clean",        0.0, None, None),
    ("hard",         1.0, None, None),
    ("heavy_shm.25", 1.0, None, {"mutation_rate": 0.25}),
    ("fragment_120", 1.0, 120,  None),
    ("fragment_80",  1.0, 80,   None),
]


def aligner():
    a = Align.PairwiseAligner(); a.mode = "local"
    a.match_score = 2.0; a.mismatch_score = -1.0
    a.open_gap_score = -3.0; a.extend_gap_score = -1.0
    return a


def zscore(d):
    vs = list(d.values())
    if len(vs) < 2:
        return {k: 0.0 for k in d}
    mu = sum(vs) / len(vs)
    sd = (sum((v - mu) ** 2 for v in vs) / len(vs)) ** 0.5 or 1.0
    return {k: (v - mu) / sd for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--calibration", default=".private/models/allele_set_calibration.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--topk", type=int, default=48)
    ap.add_argument("--pred-bounds", action="store_true",
                    help="use the MODEL's predicted V boundaries (production) instead of truth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    germ = dict(zip(rs.gene("V").names, rs.gene("V").sequences))
    ck = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ck["config"]); m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    cal = json.load(open(args.calibration)) if os.path.exists(args.calibration) else None
    aln = aligner()

    print(f"{'stratum':14s} {'learned':>8} {'raw':>8} {'blend':>8} {'igblast':>8} {'ceiling':>8}")
    for name, p, crop, ov in STRATA:
        recs = gen_records(p, args.n, args.seed, crop, overrides=ov)
        reads = [r["sequence"] for r in recs]
        igb = [igblast_to_pred(x) for x in run_igblast(recs)]
        da = predict_reads(m, rs, reads, device=device, topk=args.topk, rerank="learned",
                           calibration=cal, emit_scores=True)
        L = R = B = I = C = tot = 0
        for rec, ip, pr in zip(recs, igb, da):
            truth = rec.get("v_call")
            if not truth:
                continue
            tset = set(str(truth).split(","))
            vscores = pr.get("v_scores")          # list of (name, learned_score)
            if not vscores:
                continue
            cands = [c for c, _ in vscores]
            tot += 1
            I += (ip or {}).get("v_call") in tset
            C += bool(tset & set(cands))
            if args.pred_bounds:
                vs, ve = pr.get("v_sequence_start", 0) or 0, pr.get("v_sequence_end", 0) or 0
            else:
                vs, ve = rec.get("v_sequence_start", 0) or 0, rec.get("v_sequence_end", 0) or 0
            seg = rec["sequence"][vs:ve].upper()
            rscore = {c: aln.score(seg, germ[c]) for c in cands if c in germ}
            ld = {c: s for c, s in vscores if c in rscore}
            if not ld:
                continue
            L += (max(ld, key=ld.get) in tset)
            R += (max(rscore, key=rscore.get) in tset)
            lz, rz = zscore(ld), zscore(rscore)
            blend = {c: lz[c] + rz[c] for c in ld}
            B += (max(blend, key=blend.get) in tset)
        if tot:
            print(f"{name:14s} {L/tot:8.3f} {R/tot:8.3f} {B/tot:8.3f} {I/tot:8.3f} {C/tot:8.3f}")


if __name__ == "__main__":
    main()
