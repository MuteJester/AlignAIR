"""``alignair train`` — train a custom AlignAIR model on GenAIRR dataconfig(s)."""
from __future__ import annotations

from ..api import train_model


def register(sub) -> None:
    p = sub.add_parser("train", help="train a custom AlignAIR model on GenAIRR dataconfig(s)")
    p.add_argument("--dataconfig", nargs="+", required=True,
                   help="one or more GenAIRR dataconfigs (1 => single-chain, N => multi-chain)")
    p.add_argument("--out", required=True, help="output checkpoint path (.pt)")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--short-boost", type=int, default=1,
                   help="repeat the short-read amplicon streams N times to concentrate on short reads")
    p.add_argument("--resume", default=None, help="resume from an existing checkpoint")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.set_defaults(func=run)


def run(args) -> int:
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = train_model(args.dataconfig, out_path=args.out, steps=args.steps, device=device,
                      lr=args.lr, batch_size=args.batch_size, short_boost=args.short_boost,
                      resume_path=args.resume)
    print(f"trained -> {out}")
    return 0
