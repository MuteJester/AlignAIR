"""Real-data validation on OAS human IGH reads (no GenAIRR simulation).

Real repertoire reads have no ground truth, so we use IgBLAST's FULL-READ call as a
silver standard (IgBLAST is reliable on full reads), then test the regime we claim to win:

1. FULL-READ CONCORDANCE — agreement between DNAlignAIR and IgBLAST on full reads (sanity:
   on the easy case we should track the incumbent).
2. CROP-BACK FRAGMENT TEST — crop each read to a CDR3-centered fragment, run BOTH tools on
   the fragment, and measure how often each recovers the full-read silver-truth call. This
   is our headline claim (fragment robustness), validated on REAL data.

DNAlignAIR and our IgBLAST share the OGRDB germline set, so calls are directly comparable.
"""
import argparse
import csv
import gzip
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import run_igblast, igblast_to_pred  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402

GENES = ("v", "d", "j")


def load_oas(path, n, min_len, max_len):
    rows = []
    with gzip.open(path, "rt") as f:
        f.readline()                                  # OAS metadata JSON line
        for r in csv.DictReader(f):
            s = (r.get("sequence") or "").upper()
            if set(s) <= set("ACGTN") and min_len <= len(s) <= max_len:
                rows.append(s)
            if len(rows) >= n:
                break
    return rows


def gene_of(call):
    return call.split("*")[0] if call else None


def crop_cdr3(seq, v_end, L):
    """CDR3-centered fragment of length L (V tip + CDR3 + J start) around the V end."""
    v_end = int(v_end) if v_end is not None else int(len(seq) * 0.8)
    start = max(0, min(v_end - L // 2, len(seq) - L))
    return seq[start:start + L]


def recovers(pred, truth, gene, level="allele"):
    """Did `pred` recover the full-read silver truth for this gene? allele = top1 matches
    (or is in the predicted set); gene = gene-level match."""
    t = truth.get(f"{gene}_call")
    if not t:
        return None
    p1 = (pred or {}).get(f"{gene}_call")
    if level == "gene":
        # accept DNAlignAIR's hierarchical resolved call too (gene/family aware)
        rc = (pred or {}).get(f"{gene}_resolved_call")
        cand = {gene_of(p1)}
        if rc:
            cand.add(rc if "*" not in rc else gene_of(rc))
        return float(gene_of(t) in cand)
    pset = set((pred or {}).get(f"{gene}_call_set") or ([p1] if p1 else []))
    return float(t in pset or p1 == t)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--data", default=".private/realdata/chen2020_ighg.csv.gz")
    ap.add_argument("--calibration", default=".private/models/allele_set_calibration.json")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--crop", type=int, default=120)
    ap.add_argument("--min-len", type=int, default=280)
    ap.add_argument("--max-len", type=int, default=540)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ckpt = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DNAlignAIR(cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    import json
    cal = json.load(open(args.calibration)) if os.path.exists(args.calibration) else None

    reads = load_oas(args.data, args.n, args.min_len, args.max_len)
    print(f"loaded {len(reads)} real OAS IGHG reads | model={os.path.basename(args.model)}")

    # silver truth = IgBLAST on FULL reads; keep confident (V+J called)
    recs = [{"sequence": s} for s in reads]
    igb_full = [igblast_to_pred(r) for r in run_igblast(recs)]
    keep = [i for i, p in enumerate(igb_full) if p and p.get("v_call") and p.get("j_call")]
    reads = [reads[i] for i in keep]; truth = [igb_full[i] for i in keep]
    print(f"{len(reads)} reads with a confident IgBLAST full-read call (silver truth)\n")

    # ---- (1) full-read concordance: DNAlignAIR vs IgBLAST ----
    da_full = predict_reads(model, rs, reads, device=device, rerank="learned", calibration=cal)
    print("=== (1) FULL-READ CONCORDANCE: DNAlignAIR vs IgBLAST silver truth ===")
    for g in GENES:
        gl = [recovers(da_full[i], truth[i], g, "gene") for i in range(len(reads))]
        al = [recovers(da_full[i], truth[i], g, "allele") for i in range(len(reads))]
        gl = [x for x in gl if x is not None]; al = [x for x in al if x is not None]
        print(f"  {g.upper()}: gene-agree={sum(gl)/max(len(gl),1):.3f}  allele-agree(in-set)={sum(al)/max(len(al),1):.3f}  (n={len(gl)})")

    # ---- (2) crop-back: both tools on the fragment vs full-read truth ----
    frags = [crop_cdr3(reads[i], truth[i].get("v_sequence_end"), args.crop) for i in range(len(reads))]
    igb_frag = [igblast_to_pred(r) for r in run_igblast([{"sequence": s} for s in frags])]
    da_frag = predict_reads(model, rs, frags, device=device, rerank="learned", calibration=cal)
    print(f"\n=== (2) CROP-BACK to {args.crop}bp CDR3-centered fragments: recovery of full-read truth ===")
    print(f"  {'gene':<5}{'IgBLAST gene':>14}{'DNAlignAIR gene':>16}{'IgBLAST allele':>16}{'DNAlignAIR allele':>18}")
    for g in GENES:
        ig_g = [recovers(igb_frag[i], truth[i], g, "gene") for i in range(len(reads))]
        da_g = [recovers(da_frag[i], truth[i], g, "gene") for i in range(len(reads))]
        ig_a = [recovers(igb_frag[i], truth[i], g, "allele") for i in range(len(reads))]
        da_a = [recovers(da_frag[i], truth[i], g, "allele") for i in range(len(reads))]
        def m(x): x = [v for v in x if v is not None]; return sum(x)/max(len(x), 1)
        print(f"  {g.upper():<5}{m(ig_g):>14.3f}{m(da_g):>16.3f}{m(ig_a):>16.3f}{m(da_a):>18.3f}")


if __name__ == "__main__":
    main()
