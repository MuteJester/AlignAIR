"""``alignair analyze`` — summarize an AIRR TSV: repertoire composition + prediction QC + validation."""
from __future__ import annotations

from ..analyze import analyze_file, format_text
from ._options import add_report_output, emit


def register(sub) -> None:
    p = sub.add_parser("analyze", help="summarize an AIRR TSV (repertoire + QC + validation)")
    p.add_argument("input", help="AIRR rearrangement TSV (e.g. from `alignair predict`)")
    add_report_output(p, json_help="emit the report as JSON instead of text")
    p.set_defaults(func=run)


def run(args) -> int:
    report = analyze_file(args.input)
    if args.json:
        import json
        emit(args, json.dumps(report, indent=2))
    else:
        emit(args, format_text(report))
    return 0
