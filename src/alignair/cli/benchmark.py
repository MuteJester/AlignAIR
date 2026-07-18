"""``alignair benchmark`` — evaluate a model on freshly-generated GenAIRR reads, per stratum.

A self-contained self-check (call accuracy, coord MAE, junction_nt_exact). The full head-to-head
against IgBLAST lives in the separate ``alignair_benchmark`` package.
"""
from __future__ import annotations

import json


def register(sub) -> None:
    p = sub.add_parser("benchmark", help="evaluate a model on generated GenAIRR reads (per stratum)")
    p.add_argument("--model", required=True, help="a .alignair/.pt path or a shipped registry id")
    p.add_argument("--dataconfig", nargs="+", default=None,
                   help="GenAIRR dataconfig(s) to generate from (default: the model card's)")
    p.add_argument("--n", type=int, default=200, help="reads per stratum")
    p.add_argument("--strata", default=None, help="comma-list subset (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default=None)
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--registry", action="append")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--trust-pickle", action="store_true")
    p.set_defaults(func=run)


def _dataconfig_names(args, model_path):
    if args.dataconfig:
        return args.dataconfig
    from ..model_file import container, read_metadata
    if container.is_alignair_file(model_path):                    # fall back to the card's dataconfig(s)
        return [dc["name"] for dc in read_metadata(model_path).get("reference", {}).get("dataconfigs", [])
                if dc.get("name")]
    return []


def run(args) -> int:
    import torch

    import GenAIRR.data as gd
    from ..api import load_model
    from ..evaluate import format_text, run_benchmark
    from ..registry import resolve_model, sources as _sources
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_path = str(resolve_model(args.model, sources=_sources.resolve_sources(args.registry),
                                        offline=args.offline))
    except Exception as e:
        print(f"could not resolve model '{args.model}': {e}")
        return 1
    names = _dataconfig_names(args, model_path)
    if not names:
        print("--dataconfig is required to generate benchmark data (the model card has none)")
        return 1
    try:
        dcs = [getattr(gd, n) if isinstance(n, str) else n for n in names]
    except AttributeError as e:
        print(f"unknown GenAIRR dataconfig: {e}")
        return 1
    try:
        model, reference = load_model(model_path, dataconfigs=args.dataconfig or names, device=device,
                                      trust_pickle=args.trust_pickle)
    except ValueError as e:
        print(str(e))
        return 1

    strata = [s.strip() for s in args.strata.split(",")] if args.strata else None
    results = run_benchmark(model, reference, dcs[0], n=args.n, seed=args.seed, strata_names=strata,
                            device=device, batch_size=args.batch_size)
    text = json.dumps(results, indent=2) if args.format == "json" else format_text(results)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0
