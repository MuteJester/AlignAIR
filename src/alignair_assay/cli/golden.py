"""``golden``: record a model's AIRR rows on a fixed read set, or check that a model file under this code gives the
same rows (``diagnostics.golden``)."""
from __future__ import annotations

from alignair_assay.diagnostics.golden import TOLERANCE, check_golden, record_golden

SHOWN = 20          # the differences a failed check prints


def register(sub) -> None:
    """Add the ``golden`` subcommand, with its ``record`` and ``check`` actions."""
    p = sub.add_parser("golden", help="record a model's AIRR rows on a read set, or check another code or model file "
                                      "gives the same rows")
    actions = p.add_subparsers(dest="action", required=True)
    record = actions.add_parser("record", help="predict the reads and write the rows as the golden file")
    record.add_argument("--model", required=True, help="the .alignair model file")
    record.add_argument("--reads", required=True, help="the reads, FASTA")
    record.add_argument("--out", required=True, help="the golden file to write (JSON lines)")
    check = actions.add_parser("check", help="predict the golden file's reads and compare the rows")
    check.add_argument("--model", required=True, help="the .alignair model file")
    check.add_argument("--golden", required=True, help="a golden file written by golden record")
    check.add_argument("--tolerance", type=float, default=TOLERANCE,
                       help="how far floats may differ (default %(default)s)")
    p.set_defaults(func=run)


def run(args) -> int:
    """``record``: write the golden file and print its size. ``check``: print how many rows differ and the first
    ``SHOWN`` differences; exit 0 when none does, 1 otherwise."""
    if args.action == "record":
        n = record_golden(args.model, args.reads, args.out)
        print(f"recorded {n} rows -> {args.out}")
        return 0
    report = check_golden(args.model, args.golden, args.tolerance)
    differing = len({read_id for read_id, *_ in report.differences})
    print(f"{report.n} rows, {differing} differ")
    for read_id, key, expected, actual in report.differences[:SHOWN]:
        print(f"  {read_id} {key}: {expected!r} -> {actual!r}")
    return 0 if not report.differences else 1
