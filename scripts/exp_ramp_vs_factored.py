"""Decisive experiment: scalar ramp vs coupled mixture vs FACTORED per-axis pacing.
Trains three short identical models and reports each one's FrozenLattice competence
field. The factored arm is expected to reach the heavy_shm_fulllen corner the coupled
ramp structurally excludes — but ONLY if it gets enough advance cycles to climb, so the
curriculum is advanced every `--advance-every` steps (not just a handful of times).

Run (scale):
  PYTHONPATH=src ./.venv/bin/python scripts/exp_ramp_vs_factored.py \
      --locus igh --d-model 96 --steps 6000 --advance-every 400 --n-per-cell 400 --batch-size 32
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

_LOCUS = {"igk": gdata.HUMAN_IGK_OGRDB, "igh": gdata.HUMAN_IGH_OGRDB}


def _curriculum(arm):
    return {"ramp": Curriculum(), "mixture": StratifiedCurriculum(),
            "factored": FactoredCurriculum(start_pace=0.1)}[arm]


def run_arm(arm: str, steps: int, n_per_cell: int, batch_size: int = 16, seed: int = 0,
            device=None, locus: str = "igk", d_model: int = 64, advance_every: int = 300,
            advance_eval_n: int = 120, threshold: float = 0.7, step: float = 0.1) -> dict:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dc = _LOCUS[locus]
    cfg = DNAlignAIRConfig(d_model=d_model, n_layers=2, nhead=4, dim_feedforward=2 * d_model)
    rs = ReferenceSet.from_dataconfigs(dc)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    cur = _curriculum(arm)
    gym = AlignAIRGym([dc], rs, n=batch_size * 4, seed=seed, curriculum=cur)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=batch_size, device=device)
    lat = FrozenLattice.standard(seed=seed)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(), [dc], device=device)
    done = 0
    while done < steps:
        chunk = min(advance_every, steps - done)
        trainer.fit(total_steps=chunk, global_total=steps, progress=False)
        done += chunk
        if isinstance(cur, FactoredCurriculum):
            trainer.advance_curriculum(ev.eval_all(n_per_cell=advance_eval_n),
                                       threshold=threshold, step=step)
    return ev.eval_all(n_per_cell=n_per_cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locus", default="igk", choices=["igk", "igh"])
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--advance-every", type=int, default=300)
    ap.add_argument("--n-per-cell", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    results = {}
    for arm in ("ramp", "mixture", "factored"):
        results[arm] = run_arm(arm, args.steps, args.n_per_cell, batch_size=args.batch_size,
                               locus=args.locus, d_model=args.d_model,
                               advance_every=args.advance_every)
        print(f"\n=== {arm} ===")
        for name, v in results[arm].items():
            print(f"  {name:22s} S={v['S']:.3f}  [{v['lo']:.3f},{v['hi']:.3f}]")
    # headline: the contested corner
    print("\n--- heavy_shm_fulllen (the corner the coupled ramp excludes) ---")
    for arm in ("ramp", "mixture", "factored"):
        v = results[arm]["heavy_shm_fulllen"]
        print(f"  {arm:10s} S={v['S']:.3f}  [{v['lo']:.3f},{v['hi']:.3f}]")


if __name__ == "__main__":
    main()
