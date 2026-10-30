"""Held-out evaluation: set-aware top-1-in-set allele accuracy on a dataconfig.

Encodes the reference ONCE (spec: cache reference embeddings) then scores each read, so
periodic in-training eval is cheap.
"""
from alignair.core.runtime import evaluation_mode
from alignair.reference.reference_set import ReferenceSet
from alignair.train.gym.experiment import build_experiment
from alignair.train.gym.curriculum import Curriculum

from .config import MatcherConfig
from .data import GENES, _split_calls


def evaluate(model, dataconfig, cfg: MatcherConfig, n=200, progress=0.3, seed=0, device="cpu"):
    ref = ReferenceSet.from_dataconfigs(dataconfig)
    present = [g for g in GENES if g in ref.genes]
    hit = {g: 0 for g in present}
    tot = {g: 0 for g in present}
    params = dict(Curriculum(indel_form="uniform").params(progress))   # pinned: the eval data never drifts
    params["invert_d_prob"] = 0.0
    exp = build_experiment(dataconfig, params, allow_curatable=True)
    model = model.to(device)
    with evaluation_mode(model):
        cached = model.encode_reference(ref)
        for rec in exp.stream_records(n=n, seed=seed):
            out = model.predict_cached(cached, str(rec["sequence"]).upper())
            for g in present:
                truth = set(_split_calls(rec.get(f"{g.lower()}_call")))
                if not truth:
                    continue
                tot[g] += 1
                if truth & set(out[g]["names"]):        # set-aware: any true name in the called group
                    hit[g] += 1
    return {g: (hit[g] / tot[g] if tot[g] else 0.0) for g in present}
