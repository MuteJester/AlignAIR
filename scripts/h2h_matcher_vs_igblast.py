"""AIRRMatch vs IgBLAST head-to-head on HUMAN IGH across difficulty strata.

Both tools use the SAME OGRDB germline the reads were simulated from. Scoring is
set-aware top-1-in-set call accuracy per gene (a call is correct if the tool's predicted
allele set intersects the ground-truth allele set) — the honest metric for both, since
each emits ties as a set (IgBLAST comma-list; AIRRMatch sequence-identical group).

Inverted-D is disabled (invert_d_prob=0): AIRRMatch v1 uses one V-anchored read orientation
and does not handle independently-inverted D, so this is the scope-fair comparison.
"""
import argparse
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records, run_igblast  # noqa: E402  reuse IgBLAST harness

import GenAIRR.data as gd  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.matcher.config import MatcherConfig  # noqa: E402
from alignair.matcher.model import AIRRMatch  # noqa: E402

GENES = ("v", "d", "j")
STRATA = [
    ("clean",        0.0, None, None),
    ("moderate",     0.5, None, None),
    ("hard",         1.0, None, None),
    ("heavy_shm.25", 1.0, None, {"mutation_rate": 0.25}),
    ("fragment_120", 1.0, 120,  None),
    ("fragment_80",  1.0, 80,   None),
]


def _set(value):
    return {x for x in str(value or "").split(",") if x}


def gene_of(a):
    return a.split("*")[0] if a else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/airrmatch_diverse.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ck = torch.load(a.model, map_location=a.device)
    cfg = MatcherConfig(**ck["config"])
    model = AIRRMatch(cfg).to(a.device).eval()
    model.load_state_dict(ck["model"])
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    cached = model.encode_reference(ref)

    print(f"AIRRMatch (step {ck.get('step')}) vs IgBLAST | HUMAN_IGH_OGRDB | "
          f"n={a.n}/stratum | set-aware top-1-in-set\n")
    print(f"{'stratum':14s} {'gene':4s} {'IgBLAST':>8s} {'AIRRMatch':>9s} {'delta':>7s}")
    vloss = Counter()
    totals = {g: [0, 0, 0] for g in GENES}  # [igb_correct, am_correct, n]
    for name, p, crop, ov in STRATA:
        overrides = {"invert_d_prob": 0.0}
        if ov:
            overrides.update(ov)
        recs = gen_records(p, a.n, a.seed, crop, overrides=overrides)
        igb_rows = run_igblast(recs)
        am_preds = [model.predict_cached(cached, str(r["sequence"]).upper()) for r in recs]
        for g in GENES:
            G = g.upper()
            igb_c = am_c = tot = 0
            for rec, row, am in zip(recs, igb_rows, am_preds):
                ts = _set(rec.get(f"{g}_call"))
                if not ts:
                    continue
                tot += 1
                igb_ok = bool(_set((row or {}).get(f"{g}_call")) & ts)
                am_names = set(am.get(G, {}).get("names", [])) if G in am else set()
                am_ok = bool(am_names & ts)
                igb_c += igb_ok
                am_c += am_ok
                if g == "v" and not am_ok:
                    top = next(iter(am_names), None)
                    vloss["sibling" if gene_of(top) in {gene_of(x) for x in ts} else "diff_gene"] += 1
            if tot:
                ib, ab = igb_c / tot, am_c / tot
                print(f"{name:14s} {G:4s} {ib:8.2f} {ab:9.2f} {ab - ib:+7.2f}")
                totals[g][0] += igb_c
                totals[g][1] += am_c
                totals[g][2] += tot
        print()

    print("=== OVERALL (all strata pooled) ===")
    print(f"{'':14s} {'gene':4s} {'IgBLAST':>8s} {'AIRRMatch':>9s} {'delta':>7s}")
    for g in GENES:
        ic, ac, tot = totals[g]
        if tot:
            print(f"{'':14s} {g.upper():4s} {ic/tot:8.2f} {ac/tot:9.2f} {(ac-ic)/tot:+7.2f}")
    print(f"\nAIRRMatch V-loss cases: sibling(same-gene, discrimination)={vloss['sibling']}, "
          f"diff-gene(retrieval)={vloss['diff_gene']}")


if __name__ == "__main__":
    main()
