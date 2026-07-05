"""Decisive fast-fine-stage benchmark on HUMAN IGH: accuracy + CPU speed for
full-MaxSim vs k-mer-seed+neural vs k-mer-seed+classical-align, against IgBLAST.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records, run_igblast  # noqa: E402

import GenAIRR.data as gd  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.matcher.config import MatcherConfig  # noqa: E402
from alignair.matcher.model import AIRRMatch  # noqa: E402
from alignair.align import get_aligner  # noqa: E402

GENES = ("v", "d", "j")
STRATA = [("clean", 0.0, None, None), ("hard", 1.0, None, None),
          ("heavy_shm.25", 1.0, None, {"mutation_rate": 0.25}),
          ("fragment_120", 1.0, 120, None)]


def _set(v):
    return {x for x in str(v or "").split(",") if x}


def am_names(pred, g):
    return set(pred.get(g.upper(), {}).get("names", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/airrmatch_diverse.pt")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()

    ck = torch.load(a.model, map_location=a.device)
    cfg = MatcherConfig(**ck["config"])
    m = AIRRMatch(cfg).to(a.device).eval()
    m.load_state_dict(ck["model"])
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    cached = m.encode_reference(ref)
    aligner = get_aligner()
    torch.set_grad_enabled(False)

    methods = {
        "full_maxsim":  lambda s: m.predict_cached(cached, s),
        "seed+neural":  lambda s: m.predict_seeded(cached, s, topk=a.topk, fine="neural"),
        "seed+align":   lambda s: m.predict_seeded(cached, s, topk=a.topk, fine="align", aligner=aligner),
        "seed+HYBRID":  lambda s: m.predict_seeded(cached, s, topk=a.topk, fine="hybrid", aligner=aligner),
    }

    print(f"HUMAN_IGH_OGRDB | n={a.n}/stratum | topk={a.topk} | device={a.device} | set-aware call acc\n")
    for name, p, crop, ov in STRATA:
        overrides = {"invert_d_prob": 0.0}
        if ov:
            overrides.update(ov)
        recs = gen_records(p, a.n, a.seed, crop, overrides=overrides)
        seqs = [str(r["sequence"]).upper() for r in recs]
        igb = run_igblast(recs)
        preds = {mn: [fn(s) for s in seqs] for mn, fn in methods.items()}
        print(f"[{name}]")
        for g in GENES:
            row = {}
            for mn in methods:
                c = t = 0
                for rec, pr in zip(recs, preds[mn]):
                    ts = _set(rec.get(f"{g}_call"))
                    if not ts:
                        continue
                    t += 1
                    c += bool(am_names(pr, g) & ts)
                row[mn] = c / t if t else float("nan")
            # igblast
            ci = ti = 0
            for rec, row_ig in zip(recs, igb):
                ts = _set(rec.get(f"{g}_call"))
                if not ts:
                    continue
                ti += 1
                ci += bool(_set((row_ig or {}).get(f"{g}_call")) & ts)
            igacc = ci / ti if ti else float("nan")
            print(f"  {g.upper()}: IgBLAST={igacc:.2f}  " +
                  "  ".join(f"{mn}={row[mn]:.2f}" for mn in methods))
        print()

    # speed on CPU (clean reads, timed per method)
    recs = gen_records(0.3, a.n, 7, None, overrides={"invert_d_prob": 0.0})
    seqs = [str(r["sequence"]).upper() for r in recs]
    print("=== CPU speed (reads/s) ===")
    for mn, fn in methods.items():
        for s in seqs[:10]:
            fn(s)  # warmup
        t = time.time()
        for s in seqs:
            fn(s)
        print(f"  {mn:14s} {len(seqs)/(time.time()-t):6.1f} reads/s")
    t = time.time(); run_igblast(recs)
    print(f"  {'igblast(8thr)':14s} {len(recs)/(time.time()-t):6.1f} reads/s")


if __name__ == "__main__":
    main()
