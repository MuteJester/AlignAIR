"""Stage-1: coordinate-aware truncation training (mixed N/C/internal crops + full-length rehearsal +
AA-native subs/indel/I/L + partial-label compatible-class loss for partial V/J).

Run: PYTHONPATH=src .venv/bin/python scripts/aa/train_stage1_trunc.py --device cuda --steps 40000
"""
import argparse
import os

import torch

from alignair.aa.strategic import strategic_setup
from alignair.aa.strategic_trunc import stage1_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p-crop", type=float, default=0.4)
    ap.add_argument("--p-native", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="experiments/aa/strategic/model_stage1.pt")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    torch.manual_seed(args.seed)
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    print(f"Stage-1 V/D/J: {sum(p.numel() for p in model.parameters()):,} params; "
          f"V/J classes {meta['n_v_classes']}/{meta['n_j_classes']}; D {meta['n_d_alleles']}; "
          f"p_crop={args.p_crop} p_native={args.p_native}", flush=True)
    stage1_train(model, cfg, class_maps, meta, steps=args.steps, batch_size=args.batch_size,
                 lr=args.lr, p_crop=args.p_crop, p_native=args.p_native, seed=args.seed,
                 device=args.device, log_every=250, save_path=args.out, save_every=5000)
    print("done", flush=True)


if __name__ == "__main__":
    main()
