"""Diagnostic: localize the pointer-head accuracy regression to the READER (allele calls)
vs the COORDS (aligner) vs region. Trains a soft-DP arm and a pointer arm (same loss +
tau-anneal, only the aligner differs), then decomposes per-cell competence into its
allele / coords / region components so the regression is attributed to the responsible head.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/diag_reader_vs_coords.py \
      --locus igh --d-model 96 --steps 3000 --n-per-cell 300
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
_CELLS = ("clean", "junction_boundary", "heavy_shm_fulllen", "heavy_shm", "indel", "ambiguous")


def train_and_decompose(aligner, steps, n_per_cell, d_model, locus, batch_size, seed,
                        coord_tol, device, heartbeat_every=500):
    torch.manual_seed(seed)
    dc = _LOCUS[locus]
    cfg = DNAlignAIRConfig(d_model=d_model, n_layers=2, nhead=4, dim_feedforward=2 * d_model,
                           aligner=aligner)
    rs = ReferenceSet.from_dataconfigs(dc)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d, coord_loss="soft")
    gym = AlignAIRGym([dc], rs, n=batch_size * 8, seed=seed, curriculum=StratifiedCurriculum())
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=batch_size,
                         device=device, num_workers=0)
    done, t0 = 0, time.perf_counter()
    while done < steps:
        chunk = min(heartbeat_every, steps - done)
        trainer.fit(total_steps=chunk, global_total=steps, progress=False)
        done += chunk
        el = time.perf_counter() - t0
        print(f"[{aligner}] step {done}/{steps}  {el:.0f}s  {el/done:.2f}s/step", flush=True)
    lat = FrozenLattice.standard(seed=seed)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(coord_tol=coord_tol), [dc], device=device)
    return ev.eval_all_components(n_per_cell=n_per_cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locus", default="igh", choices=["igk", "igh"])
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-per-cell", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coord-tol", type=float, default=1.0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fields = {}
    for aligner in ("softdp", "pointer"):
        fields[aligner] = train_and_decompose(
            aligner, a.steps, a.n_per_cell, a.d_model, a.locus, a.batch_size, a.seed,
            a.coord_tol, device)
    print(f"\n=== reader(allele) vs coords vs region, soft-DP -> pointer (tol={a.coord_tol}) ===")
    print(f"{'cell':18s} {'component':9s} {'soft-DP':>8s} {'pointer':>8s} {'Δ':>8s}")
    for cell in _CELLS:
        for comp in ("allele", "coords", "region"):
            s = fields["softdp"].get(cell, {}).get(comp, {})
            p = fields["pointer"].get(cell, {}).get(comp, {})
            if not s or not p:
                continue
            d = p["S"] - s["S"]
            print(f"{cell:18s} {comp:9s} {s['S']:8.3f} {p['S']:8.3f} {d:+8.3f}")
        print()


if __name__ == "__main__":
    main()
