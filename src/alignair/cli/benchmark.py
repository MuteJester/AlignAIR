"""``alignair benchmark`` — evaluate a model on freshly-generated GenAIRR reads, per stratum.

A self-check on simulated reads (call accuracy, coordinate MAE, junction_nt_exact); it does not compare the model
with other aligners.
"""
from __future__ import annotations

import json
import sys

from ..errors import AlignAIRError, ConfigurationError, ModelError, RegistryError, report_error
from ._errors import fail


def register(sub) -> None:
    p = sub.add_parser("benchmark", help="evaluate a model on generated GenAIRR reads (per stratum)")
    p.add_argument("--model", required=True, help="a .alignair path or a shipped registry id")
    p.add_argument("--dataconfig", default=None,
                   help="the GenAIRR dataconfig to generate reads from (default: the model card's first)")
    p.add_argument("--n", type=int, default=200, help="reads per stratum")
    p.add_argument("--strata", default=None, help="comma-list subset (default: all)")
    p.add_argument("--seed", type=int, default=0, help="seed of the generated reads (default 0)")
    p.add_argument("--device", default=None, help="cpu / cuda / mps / auto (default: auto: CUDA, then Apple MPS, then CPU)")
    p.add_argument("--batch-size", type=int, default=64, help="reads per batch (default 64)")
    p.add_argument("--out", default=None, help="write the report here (default: stdout)")
    p.add_argument("--format", choices=["text", "json"], default="text", help="report format (default: text)")
    p.add_argument("--registry", action="append", help="registry url for a shipped model id (repeatable)")
    p.add_argument("--offline", action="store_true", help="never touch the network")
    p.set_defaults(func=run)


def dataconfig_name(args, model_path) -> str | None:
    """The dataconfig the benchmark generates reads from: ``--dataconfig``, else the model card's first (with a note
    on stderr when the card names several), else None."""
    if args.dataconfig:
        return args.dataconfig
    from ..model_file import container, read_metadata
    names = []
    if container.is_alignair_file(model_path):
        names = [dc["name"] for dc in read_metadata(model_path).get("reference", {}).get("dataconfigs", [])
                 if dc.get("name")]
    if len(names) > 1:
        print(f"note: the model card names {names}; generating from {names[0]} (choose with --dataconfig)",
              file=sys.stderr)
    return names[0] if names else None


def run(args) -> int:

    from ..api import load_model
    from ..core.runtime import resolve_device
    from ..evaluate import format_text, run_benchmark
    from ..genairr import resolve_dataconfig
    from ..registry import resolve_model
    from ..registry import sources as _sources
    device = resolve_device(args.device or "auto")     # as the API's device="auto": CUDA, then MPS, then CPU
    try:
        model_path = str(resolve_model(args.model, sources=_sources.resolve_sources(args.registry),
                                        offline=args.offline))
    except (AlignAIRError, OSError, ValueError) as e:
        return fail(e, error_type=RegistryError, message=f"could not resolve model '{args.model}'",
                    context={"model_id": args.model})
    name = dataconfig_name(args, model_path)
    if name is None:
        return report_error(ConfigurationError(
            "--dataconfig is required to generate benchmark data (the model card has none)"))
    dataconfig = resolve_dataconfig(name)
    try:
        model, reference = load_model(model_path, device=device)
    except ValueError as e:
        return fail(e, error_type=ModelError, context={"path": model_path})

    strata = [s.strip() for s in args.strata.split(",")] if args.strata else None
    results = run_benchmark(model, reference, dataconfig, n=args.n, seed=args.seed, strata_names=strata,
                            device=device, batch_size=args.batch_size)
    text = json.dumps(results, indent=2) if args.format == "json" else format_text(results)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0
