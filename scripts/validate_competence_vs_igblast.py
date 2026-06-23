"""Phase-1 validation gate for the hardened gym instrument.

Loads a trained DNAlignAIR checkpoint, evaluates the FrozenLattice, and prints each
cell's external competence S +/- bootstrap CI. The operator compares this competence
FIELD against the canonical IgBLAST head-to-head (scripts/run_h2h_benchmark.py +
benchmark.cli compare) to confirm the gym proxy TRACKS the real benchmark before the
competence metric is allowed to drive training (the Phase-1 success criterion).

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/validate_competence_vs_igblast.py \
      --model .private/models/scaled_long.pt --n-per-cell 2000
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.competence import CompetenceMetric
from alignair.gym.instrument.evaluator import LatticeEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--n-per-cell", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ck = torch.load(args.model, map_location=device)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"])

    lat = FrozenLattice.standard(seed=args.seed)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(), [gdata.HUMAN_IGH_OGRDB],
                          device=device)
    field = ev.eval_all(n_per_cell=args.n_per_cell)

    print(f"instrument fingerprint: {lat.fingerprint()}   (model={args.model})")
    print(f"{'cell':22s} {'S':>7} {'95% CI':>18} {'n':>7}")
    print("-" * 58)
    for name, v in field.items():
        print(f"{name:22s} {v['S']:7.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]   {v['n']:7d}")
    print("\nCompare this competence field vs scripts/run_h2h_benchmark.py to confirm "
          "the proxy tracks the IgBLAST benchmark before driving training.")


if __name__ == "__main__":
    main()
