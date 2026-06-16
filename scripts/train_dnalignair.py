"""Train DNAlignAIR on the GenAIRR gym and log convergence metrics.

Usage (from repo root):
  PYTHONPATH=src .venv/bin/python scripts/train_dnalignair.py \
      --config HUMAN_IGH_OGRDB --steps 400 --batch 16 --d-model 128 \
      --layers 4 --eval-every 50 --csv experiments/run.csv
"""
import argparse
import csv
import logging
import time

import torch

import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.training.gym_trainer import GymTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--refresh-ref-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--csv", default="experiments/run.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(args.seed)

    dc = getattr(gdata, args.config)
    rs = ReferenceSet.from_dataconfigs(dc)
    cfg = DNAlignAIRConfig(d_model=args.d_model, n_layers=args.layers, nhead=args.nhead)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([dc], rs, seed=args.seed)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=args.lr, batch_size=args.batch,
                         refresh_reference_every=args.refresh_ref_every)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"config={args.config} V={len(rs.gene('V').names)} "
          f"D={len(rs.gene('D').names) if rs.has_d else 0} J={len(rs.gene('J').names)} "
          f"| params={n_params/1e6:.2f}M | d_model={args.d_model} layers={args.layers}")

    rows = []
    t0 = time.time()
    # train in chunks so we can evaluate periodically
    step = 0
    while step < args.steps:
        chunk = min(args.eval_every, args.steps - step)
        hist = trainer.fit(total_steps=chunk)
        step += chunk
        ev = trainer.evaluate(n_batches=3)
        last = {k: sum(h[k] for h in hist[-5:]) / min(5, len(hist)) for k in hist[0]}
        dt = time.time() - t0
        row = {"step": step, "wall_s": round(dt, 1), "train_total": round(last["total"], 4),
               "eval_loss": round(ev["loss"], 4),
               "region_acc": round(ev["region_acc"], 4), "state_acc": round(ev["state_acc"], 4)}
        for g in genes:
            row[f"{g}_call"] = round(ev[f"{g}_call"], 3)
            row[f"{g}_start_dev"] = round(ev[f"{g}_start_dev"], 2)
            row[f"{g}_end_dev"] = round(ev[f"{g}_end_dev"], 2)
            row[f"{g}_gl_start_dev"] = round(ev[f"{g}_gl_start_dev"], 2)
            row[f"{g}_gl_end_dev"] = round(ev[f"{g}_gl_end_dev"], 2)
        rows.append(row)
        seg = " | ".join(
            f"{g.upper()} call={row[f'{g}_call']:.2f} "
            f"seq[{row[f'{g}_start_dev']:.1f},{row[f'{g}_end_dev']:.1f}] "
            f"gl[{row[f'{g}_gl_start_dev']:.1f},{row[f'{g}_gl_end_dev']:.1f}]" for g in genes)
        print(f"[step {step:4d}|{dt:4.0f}s] tot={row['train_total']:.2f} "
              f"region={row['region_acc']:.3f} state={row['state_acc']:.3f} || {seg}")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
