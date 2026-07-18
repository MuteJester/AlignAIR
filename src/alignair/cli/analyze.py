"""``alignair analyze`` — summarize an AIRR TSV: repertoire composition + prediction QC + validation."""
from __future__ import annotations

from ..analyze import analyze_file, format_text


def register(sub) -> None:
    p = sub.add_parser("analyze", help="summarize an AIRR TSV (repertoire + QC + validation)")
    p.add_argument("input", help="AIRR rearrangement TSV (e.g. from `alignair predict`)")
    p.add_argument("--out", default=None, help="write the report here (default: stdout)")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=run)


def run(args) -> int:
    report = analyze_file(args.input)
    if args.format == "json":
        import json
        text = json.dumps(report, indent=2)
    else:
        text = format_text(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0
