"""Aligner ablation ladder (spec §9): hold the curriculum fixed (the mixture that exposes
the contested cells) and vary ONLY the germline aligner + coordinate loss, then report each
arm's FrozenLattice competence with bootstrap CIs. The operator confirms each step climbs
>= the prior arm on heavy_shm_fulllen + junction_boundary (CI lower bound) without regressing
clean / indel / fragment.

Arms (--arm), mapping to (aligner, coord_loss, band_half_width):
  softdp_softargmax  : softdp,  soft, 0   # ablation #1 -- new loss on the existing soft-DP
  pointer_ce         : pointer, ce,   0   # ablation #2 -- both DP sites gone, CE-only (latency)
  pointer_softargmax : pointer, soft, 0   # ablations #3-5 -- pointer + new loss
  pointer_band       : pointer, soft, 6   # ablation #6 -- + indel band

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_aligner_ablation.py \
      --arm pointer_softargmax --locus igh --d-model 64 --steps 3000 --n-per-cell 200
"""
import argparse
import time

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

_LOCUS = {"igk": gdata.HUMAN_IGK_OGRDB, "igh": gdata.HUMAN_IGH_OGRDB}

_ARMS = {
    "softdp_ce": ("softdp", "ce", 0),                # baseline (ablation #1 control)
    "softdp_softargmax": ("softdp", "soft", 0),      # ablation #1 -- new loss, same aligner
    "pointer_ce": ("pointer", "ce", 0),
    "pointer_softargmax": ("pointer", "soft", 0),
    "pointer_band": ("pointer", "soft", 6),
}


def run_arm(arm: str, steps: int, n_per_cell: int, batch_size: int = 16, seed: int = 0,
            device=None, locus: str = "igh", d_model: int = 64, workers: int = 0,
            coord_tol: float = 2.0, heartbeat_every: int = 250) -> dict:
    aligner, coord_loss, band = _ARMS[arm]
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dc = _LOCUS[locus]
    cfg = DNAlignAIRConfig(d_model=d_model, n_layers=2, nhead=4, dim_feedforward=2 * d_model,
                           aligner=aligner, band_half_width=band)
    rs = ReferenceSet.from_dataconfigs(dc)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d, coord_loss=coord_loss)
    gym = AlignAIRGym([dc], rs, n=batch_size * 8, seed=seed, curriculum=StratifiedCurriculum())
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=batch_size,
                         device=device, num_workers=workers)
    # chunked fit with a heartbeat: trainer state (optimizer, _global_step) persists across
    # fit() calls (workers=0 draws fresh synthetic batches each call), so this is equivalent
    # to one fit() but emits a watchable step/rate/ETA line every `heartbeat_every` steps.
    done, t0 = 0, time.perf_counter()
    while done < steps:
        chunk = min(heartbeat_every, steps - done)
        trainer.fit(total_steps=chunk, global_total=steps, progress=False)
        done += chunk
        el = time.perf_counter() - t0
        rate = el / max(done, 1)
        eta = rate * (steps - done)
        print(f"[{arm}] step {done}/{steps}  elapsed {el:6.0f}s  {rate:5.2f}s/step  "
              f"ETA {eta:6.0f}s", flush=True)
    lat = FrozenLattice.standard(seed=seed)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(coord_tol=coord_tol), [dc], device=device)
    return ev.eval_all(n_per_cell=n_per_cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(_ARMS))
    ap.add_argument("--locus", default="igh", choices=["igk", "igh"])
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-per-cell", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--coord-tol", type=float, default=2.0)
    ap.add_argument("--heartbeat-every", type=int, default=250)
    args = ap.parse_args()
    field = run_arm(args.arm, args.steps, args.n_per_cell, batch_size=args.batch_size,
                    seed=args.seed, locus=args.locus, d_model=args.d_model, workers=args.workers,
                    coord_tol=args.coord_tol, heartbeat_every=args.heartbeat_every)
    print(f"\n=== {args.arm} (locus={args.locus}, steps={args.steps}) ===")
    for name, v in field.items():
        print(f"  {name:22s} S={v['S']:.3f}  [{v['lo']:.3f},{v['hi']:.3f}]")
    print("\n--- contested cells ---")
    for name in ("heavy_shm_fulllen", "junction_boundary"):
        if name in field:
            v = field[name]
            print(f"  {name:22s} S={v['S']:.3f}  [{v['lo']:.3f},{v['hi']:.3f}]")


if __name__ == "__main__":
    main()
