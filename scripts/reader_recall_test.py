"""Does widening the reader's candidate net fix heavy-SHM V? (fallback #4 test)

The probe showed no cheap recall score (cosine/maxsim/kmer) surfaces the true heavy-SHM V
allele reliably. But the soft-DP READER is a full-alignment scorer (~SW-level, 0.86 when
the truth is present). If the cap is purely that the reader only sees top-k from a weak
recall score, then letting it score a WIDER net should lift heavy-SHM V toward IgBLAST.

We run predict_reads(rerank="learned") at increasing topk and report per-gene call accuracy
on the same heavy-SHM records — isolating "recall into the reader" as the lever.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records, score, GENES  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_novel.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--topks", default="16,32,64,128,200")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ckpt = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DNAlignAIR(cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    topks = [int(x) for x in args.topks.split(",")]

    # heavy-SHM full reads (the one stratum we lose to IgBLAST)
    recs = gen_records(1.0, args.n, args.seed, None, overrides={"mutation_rate": 0.25})
    reads = [r["sequence"] for r in recs]
    print(f"heavy-SHM~0.25 | n={len(reads)} | model={args.model}\n")
    print(f"{'topk':>5} {'V':>6} {'D':>6} {'J':>6} {'V-set-rec':>10} {'V-set-sz':>9} {'sec':>6}")
    for k in topks:
        t0 = time.time()
        preds = predict_reads(model, rs, reads, device=device, topk=k, rerank="learned")
        s = score(recs, preds)
        dt = time.time() - t0
        print(f"{k:>5} {s['v']['call']:>6.3f} {s['d']['call']:>6.3f} {s['j']['call']:>6.3f} "
              f"{s['v']['srec']:>10.3f} {s['v']['ssize']:>9.2f} {dt:>6.1f}")


if __name__ == "__main__":
    main()
