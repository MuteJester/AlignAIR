"""Pieces shared by the commands: argument types and matching summaries. The truth reference of built suites is
``scoring.pipeline.truth_reference``; report provenance and report files are ``workflow.scoring``'s, shared with
``all``."""
from __future__ import annotations

import argparse
from pathlib import Path

from alignair_assay.runners.base import absolute_path


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


def print_matching(tool: str, m: dict) -> None:
    """One line of prediction/truth matching counts for ``tool`` (see ``scoring.pipeline.score_tool``)."""
    print(f"{tool}: {m['scored']} cases scored ({m['excluded']} excluded of {m['truth_cases']}); "
          f"{m['rows']} rows, {m['duplicates']} duplicates, {m['invalid_rows']} invalid rows; "
          f"missing {m['missing']}, extra {m['extra']}, ignored for excluded cases {m['excluded_ignored']}")

