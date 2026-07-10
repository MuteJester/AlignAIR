"""`alignair convert` — package a checkpoint into a safe, pickle-free `.alignair` for distribution.

Handles both a legacy `.pt` checkpoint and a legacy `.alignair` that predates the safe
`reference_json` section. Both read pickle (torch.load / the embedded dataconfig), so both require
`--trust-pickle`. The output always carries a safe `reference_json` and NO pickle sections.
"""
from __future__ import annotations

import torch

from .. import model_file as mf
from ..api import _remap_state_dict
from ..core import AlignAIR
from ..core.config import AlignAIRConfig
from ..model_file import container


def register(sub) -> None:
    p = sub.add_parser("convert", help="package a .pt / legacy .alignair into a safe, pickle-free .alignair")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--dataconfig", nargs="+", default=None,
                   help="GenAIRR dataconfig(s) the checkpoint was trained on; required for a .pt, "
                        "read from the file for a legacy .alignair")
    p.add_argument("--trust-pickle", action="store_true",
                   help="required consent: converting reads pickle (torch.load / embedded dataconfig)")
    p.set_defaults(func=run)


def _convert_legacy_alignair(src, dst) -> int:
    lm = mf.load_model(src, trust_pickle=True)                 # rebuild model + reference (pickle path)
    dcs = mf.read_dataconfig(src)                              # embedded dataconfig(s), pickle
    dcs = dcs if isinstance(dcs, list) else [dcs]
    md = mf.read_metadata(src)
    mf.save_model(dst, lm.model, dataconfigs=dcs, training=md.get("training", {"steps": 0, "batch_size": 0}),
                  include_trusted_pickle=False, model_id=md.get("model_id"),
                  model_version=md.get("model_version"))
    return 0


def _convert_pt(src, dst, dataconfig) -> int:
    if not dataconfig:
        print("--dataconfig is required to convert a legacy .pt (the reference the model was trained on)")
        return 1
    ck = torch.load(src, map_location="cpu", weights_only=False)   # pickle (trusted via --trust-pickle)
    cfg = AlignAIRConfig(**ck["config"])
    model = AlignAIR(cfg)
    model.load_state_dict(_remap_state_dict(ck["model"]), strict=True)
    mf.save_model(dst, model, dataconfigs=dataconfig, include_trusted_pickle=False,
                  training={"steps": int(ck.get("step", 0)), "batch_size": 0})
    return 0


def run(args) -> int:
    if not args.trust_pickle:
        print("converting reads pickle (torch.load for .pt, or the embedded dataconfig for a legacy "
              ".alignair). Pass --trust-pickle only for a source you trust.")
        return 1
    is_alignair = container.is_alignair_file(args.src)
    rc = _convert_legacy_alignair(args.src, args.dst) if is_alignair \
        else _convert_pt(args.src, args.dst, args.dataconfig)
    if rc == 0:
        print(f"converted {args.src} -> {args.dst}  (safe, pickle-free)")
    return rc
