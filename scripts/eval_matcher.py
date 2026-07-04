"""Evaluate AIRRMatch: set-aware top-1-in-set allele accuracy on a (held-out) dataconfig."""
import argparse

import torch

from alignair.reference.reference_set import ReferenceSet
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch
from alignair.matcher.data import GENES, _split_calls
from alignair.gym.experiment import build_experiment
from alignair.gym.curriculum import Curriculum


def evaluate(model, dataconfig, cfg: MatcherConfig, n=200, progress=0.3, seed=0, device="cpu"):
    ref = ReferenceSet.from_dataconfigs(dataconfig)
    present = [g for g in GENES if g in ref.genes]
    hit = {g: 0 for g in present}
    tot = {g: 0 for g in present}
    params = dict(Curriculum().params(progress))
    params["invert_d_prob"] = 0.0
    exp = build_experiment(dataconfig, params, allow_curatable=True)
    model = model.to(device).eval()
    for rec in exp.stream_records(n=n, seed=seed):
        out = model.predict(ref, str(rec["sequence"]).upper())
        for g in present:
            truth = set(_split_calls(rec.get(f"{g.lower()}_call")))
            if not truth:
                continue
            tot[g] += 1
            if truth & set(out[g]["names"]):        # set-aware: any true name in the called group
                hit[g] += 1
    return {g: (hit[g] / tot[g] if tot[g] else 0.0) for g in present}


def _resolve(name):
    import GenAIRR.data as gd
    return getattr(gd, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="HUMAN_IGH_OGRDB", help="held-out dataconfig name")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ck = torch.load(a.model, map_location=a.device)
    cfg = MatcherConfig(**ck["config"])
    model = AIRRMatch(cfg)
    model.load_state_dict(ck["model"])
    acc = evaluate(model, _resolve(a.config), cfg, n=a.n, progress=a.progress, device=a.device)
    print("set-aware allele accuracy:", {g: round(v, 3) for g, v in acc.items()})
