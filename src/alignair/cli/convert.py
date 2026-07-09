"""`alignair convert` — upgrade a legacy .pt checkpoint to a self-contained .alignair model file."""
from __future__ import annotations

import torch

from .. import model_file as mf
from ..api import _remap_state_dict
from ..core import AlignAIR
from ..core.config import AlignAIRConfig


def register(sub) -> None:
    p = sub.add_parser("convert", help="convert a legacy .pt checkpoint to .alignair")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--dataconfig", nargs="+", required=True,
                   help="GenAIRR dataconfig(s) the checkpoint was trained on (embedded into the .alignair)")
    p.set_defaults(func=run)


def run(args) -> int:
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    cfg = AlignAIRConfig(**ck["config"])
    model = AlignAIR(cfg)
    model.load_state_dict(_remap_state_dict(ck["model"]), strict=True)
    mf.save_model(args.dst, model, dataconfigs=args.dataconfig,
                  training={"steps": int(ck.get("step", 0)), "batch_size": 0})
    print(f"converted {args.src} -> {args.dst}")
    return 0
