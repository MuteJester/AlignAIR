"""Re-profile the CURRENT model (post soft-DP removal) end-to-end, soft-DP vs pointer,
at the production config (d320/10L/shared). Random weights -> identical compute cost, so
the STAGE breakdown is valid for timing. Answers: with the soft-DP gone, what is the new
bottleneck, and how much of it is precision-NEUTRAL (CPU-sync, re-encode, etc.)?

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/profile_aligners.py --n 1024 --batch-size 64
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import profile_inference  # noqa: E402
from profile_inference import Stage, staged_predict  # noqa: E402
from baseline_igblast import gen_records  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402

PROD = dict(d_model=320, n_layers=10, nhead=8, dim_feedforward=512, max_len=1024,
            backbone="shared", region_decoder="linear", caller="retrieval",
            allele_counts={"V": 198, "D": 33, "J": 7})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device == "cuda"
    profile_inference.N = a.n          # Stage.report reads this module global for ms/read

    recs = gen_records(0.5, a.n, a.seed, None)
    reads = [r["sequence"] if isinstance(r, dict) else r.sequence for r in recs]
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)

    results = {}
    for aligner in ("softdp", "pointer"):
        torch.manual_seed(0)
        cfg = DNAlignAIRConfig(aligner=aligner, **PROD)
        m = DNAlignAIR(cfg).to(device).eval()
        st = Stage(cuda)
        # warmup (build lazy layers + cudnn autotune) on a small slice, untimed
        with torch.no_grad():
            staged_predict(m, rs, reads[:a.batch_size], device, a.topk, "learned",
                           a.batch_size, Stage(cuda))
        t0 = time.perf_counter(); st.sync()
        with torch.no_grad():
            staged_predict(m, rs, reads, device, a.topk, "learned", a.batch_size, st)
        st.sync(); wall = time.perf_counter() - t0
        results[aligner] = (st, wall)
        print(f"\n############ aligner = {aligner}  (n={a.n}, {a.n/wall:.1f} reads/s) ############")
        st.report(sum(st.t.values()))

    sd, pt = results["softdp"][1], results["pointer"][1]
    print(f"\n=== end-to-end: soft-DP {a.n/sd:.1f} reads/s  vs  pointer {a.n/pt:.1f} reads/s "
          f"({sd/pt:.2f}x) ===")


if __name__ == "__main__":
    main()
