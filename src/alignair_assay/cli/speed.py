"""``speed``: time tools with the speed protocol on the S1 reads of a run directory, outside ``all``."""
from __future__ import annotations

import sys

from alignair_assay.cli.common import comma_list, tool_path, truth_reference
from alignair_assay.core.layout import RunLayout
from alignair_assay.report import speed_lines
from alignair_assay.runners import RUNNERS
from alignair_assay.runners.alignair import DEFAULT_IMAGE, DEFAULT_MODEL
from alignair_assay.runners.base import absolute_path
from alignair_assay.workflow import speed
from alignair_assay.workflow.orchestrate import RunConfig

DEFAULT_TOOLS = ("igblast", "alignair")


def register(sub) -> None:
    """Add the ``speed`` subcommand."""
    p = sub.add_parser("speed", help="time each tool's cold start, steady throughput and peak RSS on S1's reads")
    p.add_argument("--run-dir", required=True, help="a run directory with S1 built (build --suites S1); one made "
                                                    "by all is timed with all --speed instead")
    p.add_argument("--tools", type=comma_list, default=DEFAULT_TOOLS,
                   help=f"tools to time ({', '.join(name for name in RUNNERS if name != 'oracle')}; default "
                        f"{','.join(DEFAULT_TOOLS)}); alignair-local is timed as well when --alignair-checkpoint is "
                        "given")
    p.add_argument("--threads", type=int, default=8, help="N of the N-thread configurations, pinned to CPUs 0 to N-1")
    p.add_argument("--configs", type=comma_list,
                   help="time only these configurations, e.g. alignair-local_cuda (default: every one)")
    p.add_argument("--n-tiny", type=int, default=speed.N_TINY, help="reads of the tiny set (the cold start)")
    p.add_argument("--n-full", type=int, default=speed.N_FULL, help="reads of the full set")
    p.add_argument("--repeats", type=int, default=speed.REPEATS, help="runs of each set")
    p.add_argument("--igblast-dir", help="directory of an unpacked IgBLAST release (igblast)")
    p.add_argument("--alignair-image", help=f"AlignAIR Docker image (alignair; default {DEFAULT_IMAGE})")
    p.add_argument("--alignair-model", default=DEFAULT_MODEL, help="AlignAIR registry model id@version (alignair)")
    p.add_argument("--alignair-checkpoint", help="AlignAIR .alignair model file (alignair-local)")
    p.set_defaults(func=run)


def run(args) -> int:
    """Time the configurations of the given tools (``workflow.speed.speed_configs``) on S1 of the run directory,
    write ``speed/speed.json`` (no config hash: it belongs to no ``all`` configuration) and print the speed table;
    2 when a configuration failed. Input errors come first: a run directory made by ``all`` (time it with ``all
    --speed``, so the measurement is tied to its manifest), S1 not built, no tool, an unknown configuration name or
    a tool without its options."""
    layout = RunLayout(absolute_path(args.run_dir))
    if layout.manifest.is_file():
        raise ValueError(f"{layout.root} was made by all; time it with all --speed and its configuration (current "
                         "builds and runs are skipped), so the measurement belongs to its manifest")
    if not args.tools:
        raise ValueError("--tools names no tool")
    reference = truth_reference(layout, [speed.SPEED_SUITE])
    cfg = RunConfig(suites=(speed.SPEED_SUITE,), tools=args.tools, threads=args.threads,
                    igblast_dir=tool_path(args.igblast_dir), alignair_image=args.alignair_image,
                    alignair_model=args.alignair_model, alignair_checkpoint=tool_path(args.alignair_checkpoint))
    record = speed.run_speed(cfg, layout, reference, n_tiny=args.n_tiny, n_full=args.n_full, repeats=args.repeats,
                             names=args.configs)
    print("\n".join(speed_lines(record)).lstrip("\n"))
    failed = [name for name, result in record["results"].items() if "error" in result]
    for name in failed:
        print(f"error: speed {name} failed: {record['results'][name]['error']}; see "
              f"{layout.speed_work_dir(name)}", file=sys.stderr)
    return 2 if failed else 0
