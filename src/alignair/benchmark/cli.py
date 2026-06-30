"""CLI entry point for GenAIRR-backed AlignAIR benchmark tools.

Usage:
  PYTHONPATH=src python -m alignair.benchmark.cli build --out experiments/bench.jsonl
  PYTHONPATH=src python -m alignair.benchmark.cli build --recipe assay --coverage-planned --out experiments/bench.jsonl
  PYTHONPATH=src python -m alignair.benchmark.cli build-suite --out experiments/suite.jsonl
  PYTHONPATH=src python -m alignair.benchmark.cli summary experiments/bench.jsonl
"""
from __future__ import annotations

import argparse

from .commands import register_commands


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="GenAIRR-backed AlignAIR benchmark tools")
    subparsers = parser.add_subparsers(required=True)
    register_commands(subparsers)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
