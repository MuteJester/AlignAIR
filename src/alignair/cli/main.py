"""Entry point for the ``alignair`` console script: dispatch to the ``predict`` / ``train`` verbs."""
from __future__ import annotations

import argparse

from . import convert as _convert_cmd
from . import export_reference as _export_ref_cmd
from . import info as _info_cmd
from . import predict as _predict_cmd
from . import train as _train_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alignair",
                                     description="AlignAIR — a neural aligner for IG/TCR repertoires.")
    sub = parser.add_subparsers(dest="command", required=True)
    _predict_cmd.register(sub)
    _train_cmd.register(sub)
    _info_cmd.register(sub)
    _export_ref_cmd.register(sub)
    _convert_cmd.register(sub)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
