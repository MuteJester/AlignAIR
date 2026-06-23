"""Decisive experiment: scalar ramp vs coupled mixture vs FACTORED per-axis pacing.
Trains three short identical models and reports each one's FrozenLattice competence
field. The factored arm is expected to reach the heavy_shm_fulllen corner the coupled
ramp structurally excludes. Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_ramp_vs_factored.py --steps 4000
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.gym.curriculum import Curriculum, StratifiedCurriculum
from alignair.gym.factored import FactoredCurriculum
from alignair.training.gym_trainer import GymTrainer
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.competence import CompetenceMetric
from alignair.gym.instrument.evaluator import LatticeEvaluator

_DC = gdata.HUMAN_IGK_OGRDB     # V/J only -> fast for the smoke/dev run; use IGH for the real run


def _curriculum(arm):
    return {"ramp": Curriculum(), "mixture": StratifiedCurriculum(),
            "factored": FactoredCurriculum(start_pace=0.1)}[arm]


def run_arm(arm: str, steps: int, n_per_cell: int, batch_size: int = 16, seed: int = 0,
            device=None) -> dict:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(_DC)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    cur = _curriculum(arm)
    gym = AlignAIRGym([_DC], rs, n=batch_size * 4, seed=seed, curriculum=cur)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=batch_size, device=device)
    lat = FrozenLattice.standard(seed=seed)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(), [_DC], device=device)
    chunk = max(1, steps // 4)
    done = 0
    while done < steps:
        trainer.fit(total_steps=min(chunk, steps - done), global_total=steps, progress=False)
        done += chunk
        if isinstance(cur, FactoredCurriculum):
            trainer.advance_curriculum(ev.eval_all(n_per_cell=n_per_cell))
    return ev.eval_all(n_per_cell=n_per_cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-per-cell", type=int, default=500)
    args = ap.parse_args()
    for arm in ("ramp", "mixture", "factored"):
        field = run_arm(arm, args.steps, args.n_per_cell)
        print(f"\n=== {arm} ===")
        for name, v in field.items():
            print(f"  {name:22s} S={v['S']:.3f}  [{v['lo']:.3f},{v['hi']:.3f}]")


if __name__ == "__main__":
    main()
