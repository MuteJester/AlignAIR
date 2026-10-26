"""Entry point for the ``alignair`` console script: dispatch to the AlignAIR verbs.

The set of subcommands here (plus ``--version``) is the CLI's versioned compatibility contract; it is
asserted by ``tests/alignair/cli/test_cli_contract.py`` and must stay in sync with the README, CI, the
Docker health check, and ``tests/packaging/release_smoke.sh``."""
from __future__ import annotations

import argparse

from .. import __version__
from ..errors import (AlignAIRError, InternalError, OperationCancelledError, OperationIOError,
                      error_context, error_reporting, report_error)
from ..aa.product import cli as _predict_aa_cmd
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
_COMMANDS = (_predict_cmd, _predict_aa_cmd, _train_cmd, _demo_cmd, _doctor_cmd, _info_cmd, _models_cmd,
             _reference_cmd, _export_ref_cmd, _convert_cmd, _validate_airr_cmd, _compare_cmd,
             _analyze_cmd, _benchmark_cmd, _genotype_cmd)


def _add_reporting_options(parser, *, root=True):
    parser.add_argument("--debug", action="store_true", default=False if root else argparse.SUPPRESS,
                        help="include the full exception traceback on errors")
    parser.add_argument("--error-format", choices=("text", "json"),
                        default="text" if root else argparse.SUPPRESS,
                        help="runtime error format on stderr (default: text)")
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in dict.fromkeys(action.choices.values()):  # aliases share a parser
                _add_reporting_options(child, root=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alignair",
                                     description="AlignAIR — a neural aligner for IG/TCR repertoires.")
    parser.add_argument("--version", action="version", version=f"alignair {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in _COMMANDS:
        cmd.register(sub)
    _add_reporting_options(parser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    with error_context(command=args.command), error_reporting(debug=args.debug,
                                                              json_output=args.error_format == "json"):
        try:
            return args.func(args) or 0
        except AlignAIRError as error:
            return report_error(error)
        except KeyboardInterrupt as exc:
            error = OperationCancelledError("Operation interrupted.")
            error.__cause__ = exc
            return report_error(error)
        except OSError as exc:
            error = OperationIOError("A file or network operation failed.",
                                     context={"errno": exc.errno, "path": exc.filename,
                                              "exception_type": type(exc).__name__},
                                     hint="Check input/output paths, permissions, and connectivity; use --debug for details.")
            error.__cause__ = exc
            return report_error(error)
        except Exception as exc:
            # The application boundary alone classifies unexpected defects. Library
            # callers still receive the original exception and traceback.
            error = InternalError("An unexpected failure interrupted the operation.",
                                  context={"exception_type": type(exc).__name__},
                                  hint="Re-run with --debug and include the traceback in an issue report.")
            error.__cause__ = exc
            return report_error(error)


if __name__ == "__main__":
    raise SystemExit(main())
