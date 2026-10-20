"""``report``: re-score, re-compare and rewrite the run report of a run directory made by ``all``."""
from __future__ import annotations

import dataclasses

from alignair_assay.cli.common import file_name
from alignair_assay.core.layout import RunLayout
from alignair_assay.runners.base import absolute_path
from alignair_assay.workflow.orchestrate import read_run_config, score_and_report


def register(sub) -> None:
    """Add the ``report`` subcommand."""
    p = sub.add_parser("report", help="score, compare and write the run report of a run directory made by all")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--a", type=file_name, help="tool a (default: the run's first tool)")
    p.add_argument("--b", type=file_name, help="tool b (default: the run's second tool)")
    p.set_defaults(func=run)


def run(args) -> int:
    """Score and compare the existing runs of the run directory's configuration (``manifest.json``), with tools
    ``--a`` and ``--b`` in place of the run's when given, and rewrite ``report.json`` and ``report.md``; nothing
    is built or run. 2 when a run failed, is missing or stale, or the oracle is not perfect."""
    layout = RunLayout(absolute_path(args.run_dir))
    cfg = read_run_config(layout)
    tools = (args.a or cfg.tools[0], args.b or cfg.tools[1])
    if tools[0] == tools[1]:
        raise ValueError(f"--a and --b must differ, both are {tools[0]!r}")
    return score_and_report(dataclasses.replace(cfg, tools=tools), layout, command="report")
