"""Train the strategic AA IGH model: explicit protein-equivalence class targets on the real gym +
AA-native substitution augmentation, trained longer.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/train_strategic_igh.py --device cuda --steps 50000
"""
import argparse
import os

import torch

from alignair.aa.strategic import strategic_setup, strategic_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sub-prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="experiments/aa/strategic/model.pt")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    torch.manual_seed(args.seed)
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    print(f"strategic AA model: {sum(p.numel() for p in model.parameters()):,} params; "
          f"V/J protein classes {meta['n_v_classes']}/{meta['n_j_classes']}; D alleles "
          f"{meta['n_d_alleles']}; sub_aug p={args.sub_prob}", flush=True)
    strategic_train(model, cfg, class_maps, d_index=meta["d_index"], steps=args.steps,
                    batch_size=args.batch_size, lr=args.lr, sub_prob=args.sub_prob, seed=args.seed,
                    device=args.device, log_every=250, save_path=args.out, save_every=5000)
    print("done", flush=True)


if __name__ == "__main__":
    main()
