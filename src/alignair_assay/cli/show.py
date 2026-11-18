"""``show``: why did this read break? One simulated case or real read, every tool side by side
(``workflow.show``, rendered by ``report.show``); a pure lookup that writes nothing."""
from __future__ import annotations

from alignair_assay.cli.common import comma_list
from alignair_assay.core.layout import RunLayout
from alignair_assay.report.show import show_markdown
from alignair_assay.runners.base import absolute_path
from alignair_assay.workflow.show import show_case, show_read

USAGE = "give --case ID, or --donor D and --read ID"


def register(sub) -> None:
    """Add the ``show`` subcommand."""
    p = sub.add_parser("show", help="one read side by side: its case or read, truth, each tool's AIRR row and "
                                    "scores, and each diagnosed model's stored scores (writes nothing)")
    p.add_argument("--run-dir", required=True, help="a run directory made by all, build + diagnose, or real")
    p.add_argument("--case", help="a simulated case id, e.g. S1:full_igm:000000")
    p.add_argument("--donor", help="a real donor, e.g. W-97 (with --read)")
    p.add_argument("--read", help="a real read id of --donor")
    p.add_argument("--tools", type=comma_list, help="comma-separated tools to show (default: every one that ran on "
                                                    "the case's suite or the donor, the oracle left out)")
    p.add_argument("--arm", default="shared", help="the arm the tools ran on (default shared)")
    p.set_defaults(func=run)


def run(args) -> int:
    """Look the case (``--case``) or the real read (``--donor`` and ``--read``) up and print it. An unknown id exits
    with status 2 naming the nearest ones, as does a tool that did not run or neither form of id."""
    layout = RunLayout(absolute_path(args.run_dir))
    tools = None if args.tools is None else list(args.tools)
    if args.case is not None and args.donor is None and args.read is None:
        found = show_case(layout, args.case, tools=tools, arm=args.arm)
    elif args.case is None and args.donor is not None and args.read is not None:
        found = show_read(layout, args.donor, args.read, tools=tools, arm=args.arm)
    else:
        raise ValueError(USAGE)
    print(show_markdown(found), end="")
    return 0
