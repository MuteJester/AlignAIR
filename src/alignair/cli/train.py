"""``alignair train`` — train a custom AlignAIR model on GenAIRR dataconfig(s)."""
from __future__ import annotations

from ..api import train_model


def register(sub) -> None:
    p = sub.add_parser("train", help="train a custom AlignAIR model on GenAIRR dataconfig(s)")
    p.add_argument("--dataconfig", nargs="+", required=True,
                   help="one or more GenAIRR dataconfigs (1 => single-chain, N => multi-chain)")
    p.add_argument("--out", required=True, help="output model path (.alignair)")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--short-boost", type=int, default=1,
                   help="repeat the short-read amplicon streams N times to concentrate on short reads")
    p.add_argument("--grad-clip", type=float, default=None,
                   help="clip the global gradient norm to this value (off by default)")
    p.add_argument("--val-every", type=int, default=0,
                   help="run a fixed-seed validation pass every N steps and keep the best checkpoint "
                        "(<out>.best.alignair); 0 disables")
    p.add_argument("--resume", default=None, help="resume from an existing .alignair checkpoint")
    p.add_argument("--resume-trust-pickle", action="store_true",
                   help="allow resuming a legacy .pt checkpoint (runs arbitrary-code pickle)")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.set_defaults(func=run)


def run(args) -> int:
    import torch
    from ..train.guards import TrainingConfigError
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        out = train_model(args.dataconfig, out_path=args.out, steps=args.steps, device=device,
                          lr=args.lr, batch_size=args.batch_size, short_boost=args.short_boost,
                          grad_clip=args.grad_clip, val_every=args.val_every,
                          resume_path=args.resume, resume_trust_pickle=args.resume_trust_pickle)
    except TrainingConfigError as e:
        print(str(e))
        return 1
    print(f"trained -> {out}")
    return 0
