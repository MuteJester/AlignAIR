"""NARROW pre-retrain sanity check (user gate): briefly train a SMALL seed_extend model from
scratch and confirm the integrated pipeline (shared encoder + band head + banded DP + retrieval)
learns coords + alleles ABOVE CHANCE on the frozen lattice — before committing to a full retrain.

Green if clean coord competence is clearly above chance and the contested cells are non-trivial.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_seed_extend_sanity.py --steps 1500 --d-model 64 --n 200
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.gym.curriculum import StratifiedCurriculum
from alignair.training.gym_trainer import GymTrainer
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.competence import CompetenceMetric
from alignair.gym.instrument.evaluator import LatticeEvaluator

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--coord-tol", type=float, default=2.0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = gdata.HUMAN_IGH_OGRDB
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=a.d_model, n_layers=2, nhead=4, dim_feedforward=2 * a.d_model,
                           backbone="shared", aligner="seed_extend")
    rs = ReferenceSet.from_dataconfigs(dc)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([dc], rs, n=a.batch_size * 8, seed=0, curriculum=StratifiedCurriculum())
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=a.batch_size, device=device)
    done = 0
    while done < a.steps:
        chunk = min(300, a.steps - done)
        trainer.fit(total_steps=chunk, global_total=a.steps, progress=False)
        done += chunk
        print(f"[train] {done}/{a.steps}", flush=True)

    lat = FrozenLattice.standard(seed=0)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(coord_tol=a.coord_tol), [dc], device=device)
    field = ev.eval_all(n_per_cell=a.n)
    print(f"\nseed_extend SANITY (d={a.d_model}, {a.steps} steps) | competence per cell (tol={a.coord_tol})")
    print(f"{'cell':18s} {'S':>8s} {'lo':>8s} {'hi':>8s}")
    for cname in CELLS:
        v = field.get(cname, {})
        print(f"{cname:18s} {v.get('S', float('nan')):8.3f} {v.get('lo', 0):8.3f} {v.get('hi', 0):8.3f}")
    print("\nGREEN if clean S is clearly above chance (>~0.5) and contested cells are non-trivial.")


if __name__ == "__main__":
    main()
