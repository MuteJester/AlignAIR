"""Entry point for the ``alignair`` console script: dispatch to the AlignAIR verbs.

The set of subcommands here (plus ``--version``) is the CLI's versioned compatibility contract; it is
asserted by ``tests/alignair/cli/test_cli_contract.py`` and must stay in sync with the README, CI, the
Docker health check, and ``scripts/release_smoke.sh``."""
from __future__ import annotations

import argparse

from .. import __version__
from . import analyze as _analyze_cmd
from . import benchmark as _benchmark_cmd
from . import compare as _compare_cmd
from . import convert as _convert_cmd
from . import demo as _demo_cmd
from . import doctor as _doctor_cmd
from . import genotype as _genotype_cmd
from . import export_reference as _export_ref_cmd
from . import info as _info_cmd
from . import models as _models_cmd
from . import predict as _predict_cmd
from . import reference as _reference_cmd
from . import train as _train_cmd
from . import validate_airr as _validate_airr_cmd

# every subcommand registrar, in help order. Keep this the single source of truth for the surface.
_COMMANDS = (_predict_cmd, _train_cmd, _demo_cmd, _doctor_cmd, _info_cmd, _models_cmd, _reference_cmd,
             _export_ref_cmd, _convert_cmd, _validate_airr_cmd, _compare_cmd, _analyze_cmd,
             _benchmark_cmd, _genotype_cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alignair",
                                     description="AlignAIR — a neural aligner for IG/TCR repertoires.")
    parser.add_argument("--version", action="version", version=f"alignair {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in _COMMANDS:
        cmd.register(sub)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
