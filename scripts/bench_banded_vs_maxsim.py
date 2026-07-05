"""Does the banded-alignment scorer beat MaxSim? Compare, on the SAME (maxsim-trained)
encoder, neural-MaxSim vs neural-Banded vs classical-align (target) vs IgBLAST, per stratum.
Banded params are untrained here (raw base-match channel does the SNP work) -- a floor.
"""
import argparse
import os
import sys

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


def names(pred, g):
    return set(pred.get(g.upper(), {}).get("names", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/airrmatch_diverse.pt")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ck = torch.load(a.model, map_location=a.device)
    base = ck["config"]
    m_max = AIRRMatch(MatcherConfig(**{**base, "scorer": "maxsim"})).to(a.device).eval()
    m_max.load_state_dict(ck["model"], strict=False)
    m_band = AIRRMatch(MatcherConfig(**{**base, "scorer": "banded"})).to(a.device).eval()
    miss, unexp = m_band.load_state_dict(ck["model"], strict=False)
    print(f"banded model: {len(miss)} new (untrained) params: {miss}")
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    cache_max = m_max.encode_reference(ref)
    cache_band = m_band.encode_reference(ref)
    aligner = get_aligner()
    torch.set_grad_enabled(False)

    methods = {
        "maxsim":  lambda s: m_max.predict_seeded(cache_max, s, fine="neural"),
        "banded":  lambda s: m_band.predict_seeded(cache_band, s, fine="neural"),
        "align":   lambda s: m_max.predict_seeded(cache_max, s, fine="align", aligner=aligner),
    }

    print(f"HUMAN_IGH | n={a.n}/stratum | neural-MaxSim vs neural-Banded(untrained) vs align vs IgBLAST\n")
    for name, p, crop, ov in STRATA:
        overrides = {"invert_d_prob": 0.0}
        if ov:
            overrides.update(ov)
        recs = gen_records(p, a.n, 123, crop, overrides=overrides)
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
                    c += bool(names(pr, g) & ts)
                row[mn] = c / t if t else float("nan")
            ci = ti = 0
            for rec, r in zip(recs, igb):
                ts = _set(rec.get(f"{g}_call"))
                if not ts:
                    continue
                ti += 1
                ci += bool(_set((r or {}).get(f"{g}_call")) & ts)
            print(f"  {g.upper()}: IgBLAST={ci/ti:.2f}  " + "  ".join(f"{mn}={row[mn]:.2f}" for mn in methods))
        print()


if __name__ == "__main__":
    main()
