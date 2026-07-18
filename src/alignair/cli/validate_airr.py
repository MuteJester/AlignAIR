"""``alignair validate-airr`` — check an AIRR rearrangement TSV for structural + coordinate soundness.

Exits non-zero if required columns are missing or any row violates a coordinate/CIGAR invariant, so it
can gate a pipeline or a release. See :mod:`alignair.io.airr_validate`.
"""
from __future__ import annotations


def register(sub) -> None:
    p = sub.add_parser("validate-airr", help="validate an AIRR TSV (columns + coordinate/CIGAR bounds)")
    p.add_argument("input", help="AIRR rearrangement TSV to validate")
    p.add_argument("--max-show", type=int, default=20, help="max individual errors to print")
    p.set_defaults(func=run)


def run(args) -> int:
    from ..io.airr_validate import validate_airr_file
    report = validate_airr_file(args.input)
    if report["missing_columns"]:
        print(f"INVALID: {args.input} is missing required AIRR columns: {report['missing_columns']}")
        return 1
    errors = report["errors"]
    if not errors:
        print(f"OK: {args.input} — {report['n_rows']} rows, no coordinate/CIGAR violations")
        return 0
    print(f"INVALID: {args.input} — {len(errors)} violation(s) across {report['n_rows']} rows:")
    for sid, msg in errors[: args.max_show]:
        print(f"  {sid}: {msg}")
    if len(errors) > args.max_show:
        print(f"  ... and {len(errors) - args.max_show} more")
    return 1
