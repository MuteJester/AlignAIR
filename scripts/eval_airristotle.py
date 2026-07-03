"""Evaluate a trained AIRRistotle v2: generate gym reads, align each via constrained decode, and
score per-gene set-aware calling accuracy. `--heldout-frac` splits accuracy into train vs held-out
alleles (the novel-allele generalization test); the coarse filter still sees the full reference so
held-out alleles remain retrievable, they were just never a training target.
"""
import argparse
import torch
import GenAIRR.data as gdata

from alignair.reference.reference_set import ReferenceSet
from alignair.gym.gym import build_experiment
from alignair.gym.curriculum import Curriculum
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle
from alignair.airristotle.data import make_retrievers
from alignair.airristotle.infer import align, called_names

GENES = ["V", "D", "J"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--v-shortlist", type=int, default=16)
    ap.add_argument("--heldout-frac", type=float, default=0.0)
    ap.add_argument("--progress", type=float, default=0.3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AIRRTokenizer()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    r = make_retrievers(rs)

    ck = torch.load(a.ckpt, map_location=dev)
    cfg = AIRRConfig(**{**ck["config"], "vocab_size": tok.vocab_size})
    m = AIRRistotle(cfg).to(dev).eval()
    m.load_state_dict(ck["model"])

    step = max(int(round(1 / a.heldout_frac)), 2) if a.heldout_frac > 0 else 0
    held = {G: set(range(0, len(rs.gene(G).names), step)) if step else set() for G in GENES}

    hit = {(G, s): 0 for G in GENES for s in ("train", "held")}
    tot = {(G, s): 0 for G in GENES for s in ("train", "held")}
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, dict(Curriculum().params(a.progress)))
    for rec in exp.stream_records(n=a.n, seed=1):
        called = called_names(align(m, str(rec["sequence"]), rs, tok, r, a.v_shortlist), rs)
        for G in GENES:
            names = [n for n in str(rec.get(f"{G.lower()}_call", "")).split(",") if n]
            if not names:
                continue
            prim = rs.gene(G).index.get(names[0])
            split = "held" if (prim is not None and prim in held[G]) else "train"
            tot[(G, split)] += 1
            if any(n in called[G] for n in names):          # set-aware: any true allele called
                hit[(G, split)] += 1

    print(f"AIRRistotle v2 eval  n={a.n}  v_shortlist={a.v_shortlist}  heldout_frac={a.heldout_frac}")
    for G in GENES:
        row = "  ".join(f"{s}={hit[(G, s)]}/{tot[(G, s)]}"
                        f"({hit[(G, s)]/tot[(G, s)]:.3f})" if tot[(G, s)] else f"{s}=-"
                        for s in ("train", "held"))
        print(f"  {G}: {row}")


if __name__ == "__main__":
    main()
