"""Pieces shared by the commands: argument types and matching summaries. The truth reference of built suites is
``scoring.pipeline.truth_reference``; report provenance and report files are ``workflow.scoring``'s, shared with
``all``."""
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from alignair_assay.runners.base import absolute_path
from alignair_assay.runners.variants import check_label


def file_name(value: str) -> str:
    """Argument type for a name used as a file-name component (a tool name): no directory part."""
    if not value or Path(value).name != value:
        raise argparse.ArgumentTypeError(f"not a plain file-name component: {value!r}")
    return value


def tool_path(path: str | None) -> str | None:
    """A tool path option (``--igblast-dir``, ``--alignair-checkpoint``) made absolute, ``~`` expanded
    (``runners.base.absolute_path``), so a recorded configuration never depends on where a command was started;
    None stays None."""
    return None if path is None else absolute_path(path)


def comma_list(value: str) -> tuple[str, ...]:
    """The non-empty, stripped items of a comma-separated list, in order and without repeats."""
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def checkpoint_spec(value: str) -> tuple[str, str]:
    """Argument type of ``--checkpoint LABEL=PATH``: the label (a checkpoint label, ``runners.variants.LABEL``) and
    the path made absolute, ``~`` expanded; the path may itself hold ``=``. ArgumentTypeError otherwise."""
    label, equals, path = value.partition("=")
    if not equals or not path:
        raise argparse.ArgumentTypeError(f"--checkpoint takes LABEL=PATH, got {value!r}")
    try:
        check_label(label)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return label, absolute_path(path)


def add_checkpoint_option(parser: argparse.ArgumentParser, tools: str) -> None:
    """Add ``--checkpoint LABEL=PATH``, repeatable, as ``args.checkpoints`` (a list of ``checkpoint_spec`` pairs;
    ``checkpoint_map`` makes it the label -> path mapping); ``tools`` names the variants it serves."""
    parser.add_argument("--checkpoint", dest="checkpoints", metavar="LABEL=PATH", action="append",
                        type=checkpoint_spec, default=[],
                        help=f"a .alignair checkpoint run as {tools}@LABEL (repeatable: one per label), e.g. "
                             "--checkpoint base=old.alignair --checkpoint cand=new.alignair")


def checkpoint_map(pairs: Iterable[tuple[str, str]] | None) -> dict[str, str]:
    """The ``--checkpoint`` pairs as label -> path, in order; ValueError for a label given twice."""
    found: dict[str, str] = {}
    for label, path in pairs or ():
        if label in found:
            raise ValueError(f"--checkpoint label {label!r} is given twice ({found[label]} and {path})")
        found[label] = path
    return found


def print_matching(tool: str, m: dict) -> None:
    """One line of prediction/truth matching counts for ``tool`` (see ``scoring.pipeline.score_tool``)."""
    print(f"{tool}: {m['scored']} cases scored ({m['excluded']} excluded of {m['truth_cases']}); "
          f"{m['rows']} rows, {m['duplicates']} duplicates, {m['invalid_rows']} invalid rows; "
          f"missing {m['missing']}, extra {m['extra']}, ignored for excluded cases {m['excluded_ignored']}")

