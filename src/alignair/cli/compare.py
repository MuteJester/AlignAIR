"""``alignair compare`` — tool-to-tool AIRR agreement between two rearrangement TSVs.

Wraps :mod:`alignair.analyze.compare`: matches rows by ``sequence_id`` and reports per-gene call agreement,
junction/productivity concordance, and equivalence-set rescue. Works on the user's own data (no ground
truth), e.g. AlignAIR vs IgBLAST/MiXCR ``exportAirr`` output, or two AlignAIR runs.
"""
from __future__ import annotations

from ._options import add_report_output, emit


def register(sub) -> None:
    p = sub.add_parser("compare", help="compare two AIRR TSVs (per-gene agreement + set rescue)")
    p.add_argument("--a", required=True, help="first AIRR rearrangement TSV")
    p.add_argument("--b", required=True, help="second AIRR rearrangement TSV")
    p.add_argument("--a-name", default="model_a", help="label for the first file")
    p.add_argument("--b-name", default="model_b", help="label for the second file")
    add_report_output(p, json_help="emit the raw comparison JSON instead of markdown")
    p.set_defaults(func=run)


def run(args) -> int:
    from ..analyze.compare import compare_airr, format_report_md, read_airr
    a, b = read_airr(args.a), read_airr(args.b)
    result = compare_airr(a, b, a_name=args.a_name, b_name=args.b_name)
    if args.json:
        import json
        emit(args, json.dumps(result, indent=2))
    else:
        emit(args, format_report_md(result))
    return 0
