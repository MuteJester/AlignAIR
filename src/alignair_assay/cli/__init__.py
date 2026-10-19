"""Command-line entry point: build suites, build observable truth, score one tool, compare two tools.

  python -m alignair_assay build --suites S0,S1 --run-dir RUN [--split dev] [--reference HUMAN_IGH_OGRDB]
      [--seed 20261019] [--scale 1.0] [--workers 8]
  python -m alignair_assay truth --bench-cases CASES.jsonl[.gz] --reference HUMAN_IGH_OGRDB --out TRUTH.jsonl.gz
  python -m alignair_assay score --truth TRUTH --predictions AIRR.tsv --tool NAME --out-dir DIR
  python -m alignair_assay compare --truth TRUTH --a A.tsv --a-name A --b B.tsv --b-name B --out-dir DIR

Predictions are matched to truth by AIRR ``sequence_id`` == truth ``case_id``. Excluded truths are not scored and
their predictions are ignored; a truth case without a prediction is scored as missing (never dropped). A prediction
file matching no scored case is an error (exit status 2), as are missing files and a truth file built against
another reference.

Each command is one module of this package with ``register(subparsers)``, which adds its subparser and sets
``func``, and ``run(args) -> int``.
"""
from __future__ import annotations

import argparse
import sys

from alignair_assay.cli import build, compare, score, truth

COMMANDS = (build, truth, score, compare)


def main(argv: list[str] | None = None) -> int:
    """Run a command; a missing file or invalid input prints ``error: ...`` and returns 2."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    """The ``python -m alignair_assay`` parser with one subcommand per module in ``COMMANDS``."""
    parser = argparse.ArgumentParser(prog="python -m alignair_assay",
                                     description="Observable-truth assay for immune-receptor aligners")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command.register(sub)
    return parser
