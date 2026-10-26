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
    """Score and compare the existing runs of the run directory's configuration (``manifest.json``) and rewrite
    ``report.json`` and ``report.md``; nothing is built or run. ``--a`` and ``--b`` may swap the run's two tools;
    another tool is refused, since the manifest records no settings to check its runs against. 2 when a run failed,
    is missing or stale (made with other tool settings, on changed files or on a suite whose build no longer
    matches), or the oracle is not perfect."""
    layout = RunLayout(absolute_path(args.run_dir))
    cfg = read_run_config(layout)
    tools = (args.a or cfg.tools[0], args.b or cfg.tools[1])
    if tools[0] == tools[1]:
        raise ValueError(f"--a and --b must differ, both are {tools[0]!r}")
    if set(tools) != set(cfg.tools):
        raise ValueError(f"{layout.root} compares {' and '.join(cfg.tools)}; report can only swap them, got "
                         f"{' and '.join(tools)} (run all with the other tools)")
    return score_and_report(dataclasses.replace(cfg, tools=tools), layout, command="report")
