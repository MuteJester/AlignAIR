"""Evaluate AIRRistotle v2 across references.

For each requested config, generate gym reads, align each via constrained decode against THAT config's
reference, and score per-gene set-aware calling accuracy. Point `--species` at the HELD-OUT species
(never seen in training) for the definitive novel-reference generalization test; the coarse filter
still sees the config's full reference, so nothing is memorized — only the in-context skill is tested.
"""
import argparse
import torch

from alignair.reference.reference_set import ReferenceSet
from alignair.gym.gym import build_experiment
from alignair.gym.curriculum import Curriculum
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle
from alignair.airristotle.data import make_retrievers
from alignair.airristotle.infer import align, called_names
from alignair.airristotle.corpus import select_configs


@torch.no_grad()
def eval_config(model, dc, tok, n, v_shortlist, progress):
    rs = ReferenceSet.from_dataconfigs(dc)
    r = make_retrievers(rs)
    genes = ["V", "J"] + (["D"] if rs.has_d else [])
    hit = {G: 0 for G in genes}
    tot = {G: 0 for G in genes}
    exp = build_experiment(dc, dict(Curriculum().params(progress)))
    for rec in exp.stream_records(n=n, seed=1):
        called = called_names(align(model, str(rec["sequence"]), rs, tok, r, v_shortlist, rs.has_d), rs)
        for G in genes:
            names = [x for x in str(rec.get(f"{G.lower()}_call", "")).split(",") if x]
            if not names:
                continue
            tot[G] += 1
            if any(x in called.get(G, []) for x in names):     # set-aware
                hit[G] += 1
    return {G: (hit[G] / tot[G] if tot[G] else float("nan")) for G in genes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--species", default="all", help="'all' or comma-list (e.g. the held-out species)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--v-shortlist", type=int, default=16)
    ap.add_argument("--progress", type=float, default=0.3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AIRRTokenizer()
    ck = torch.load(a.ckpt, map_location=dev)
    cfg = AIRRConfig(**{**ck["config"], "vocab_size": tok.vocab_size})
    m = AIRRistotle(cfg).to(dev).eval()
    m.load_state_dict(ck["model"])

    species = None if a.species == "all" else a.species.split(",")
    configs = select_configs(species=species)
    print(f"AIRRistotle v2 eval  n={a.n}/config  v_shortlist={a.v_shortlist}  {len(configs)} references")
    for name, dc in sorted(configs.items()):
        acc = eval_config(m, dc, tok, a.n, a.v_shortlist, a.progress)
        print(f"  {name:28s} " + "  ".join(f"{G}={acc[G]:.3f}" for G in ("V", "J") if G in acc)
              + (f"  D={acc['D']:.3f}" if "D" in acc else ""))


if __name__ == "__main__":
    main()
