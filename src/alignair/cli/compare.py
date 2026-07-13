"""``alignair compare`` — tool-to-tool AIRR agreement between two rearrangement TSVs.

Wraps :mod:`alignair.compare`: matches rows by ``sequence_id`` and reports per-gene call agreement,
junction/productivity concordance, and equivalence-set rescue. Works on the user's own data (no ground
truth), e.g. AlignAIR vs IgBLAST/MiXCR ``exportAirr`` output, or two AlignAIR runs.
"""
from __future__ import annotations


def register(sub) -> None:
    p = sub.add_parser("compare", help="compare two AIRR TSVs (per-gene agreement + set rescue)")
    p.add_argument("--a", required=True, help="first AIRR rearrangement TSV")
    p.add_argument("--b", required=True, help="second AIRR rearrangement TSV")
    p.add_argument("--a-name", default="model_a", help="label for the first file")
    p.add_argument("--b-name", default="model_b", help="label for the second file")
    p.add_argument("--out", default=None, help="write the markdown report here (default: stdout)")
    p.add_argument("--json", action="store_true", help="emit the raw comparison JSON instead of markdown")
    p.set_defaults(func=run)


def run(args) -> int:
    from ..compare import compare_airr, format_report_md, read_airr
    a, b = read_airr(args.a), read_airr(args.b)
    result = compare_airr(a, b, a_name=args.a_name, b_name=args.b_name)
    if args.json:
        import json
        text = json.dumps(result, indent=2)
    else:
        text = format_report_md(result)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0
