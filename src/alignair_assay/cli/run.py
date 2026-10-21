"""``run``: run one tool on built suites of a run directory, against the shared reference."""
from __future__ import annotations

import sys

from alignair_assay.cli.common import comma_list
from alignair_assay.core.layout import RunLayout
from alignair_assay.runners import RUNNERS, make_runner, parse_tool
from alignair_assay.runners.alignair import DEFAULT_IMAGE, DEFAULT_MODEL
from alignair_assay.runners.arms import run_genotype_arm
from alignair_assay.runners.base import RunnerOptions, RunResult, absolute_path, run_suite
from alignair_assay.runners.reference import ensure_shared_reference
from alignair_assay.scoring.pipeline import truth_reference

ARMS = ("shared", "genotype")
# Where each checkpoint tool's variants (``<tool>@<label>``, with ``--checkpoint LABEL=PATH``) run instead.
VARIANT_COMMANDS = {"alignair-local": "all", "alignair-api": "all or real"}


def register(sub) -> None:
    """Add the ``run`` subcommand."""
    p = sub.add_parser("run", help="run one tool on built suites of a run directory")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tool", required=True, help=f"registered tool name ({', '.join(RUNNERS)})")
    p.add_argument("--suites", required=True, help="comma-separated built suite names, e.g. S0,S1")
    p.add_argument("--arm", choices=ARMS, default="shared",
                   help="shared: every tool gets the shared reference; genotype: each donor's reads, with the tool "
                        "restricted to the donor's genotype (suites with donors only)")
    p.add_argument("--igblast-dir", help="directory of an unpacked IgBLAST release (igblast)")
    p.add_argument("--threads", type=int, default=8, help="tool threads")
    p.add_argument("--cpus", help="CPU list to pin the tool to, e.g. 0-7 (taskset; docker --cpuset-cpus)")
    p.add_argument("--image", help=f"AlignAIR Docker image (alignair; default {DEFAULT_IMAGE})")
    p.add_argument("--model", default=DEFAULT_MODEL, help="AlignAIR registry model id@version (alignair)")
    p.add_argument("--cache-dir", help="AlignAIR model cache mounted into the container (alignair; default "
                                       "runs/alignair/_work/cache, seeded from the user's cache)")
    p.add_argument("--checkpoint", help="AlignAIR .alignair model file (alignair-local, alignair-api)")
    p.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda (alignair-local, alignair-api)")
    p.add_argument("--batch-size", type=int, default=64, help="reads per batch (alignair-local, alignair-api)")
    p.set_defaults(func=run)


def run(args) -> int:
    """Run the tool on every named suite and print one line per run; 2 when any run failed.

    Every input is checked first: the tool name (a plain tool: a checkpoint variant ``<tool>@<label>`` needs the
    ``--checkpoint LABEL=PATH`` that ``all`` takes, and ``real`` for alignair-api's: ``VARIANT_COMMANDS``) and
    options, that each suite is built against one
    truth reference, and, for the genotype arm, that each suite has donors. The run directory and the tool's paths are
    made absolute, so the recorded commands can be re-run. The shared reference is then verified against (or
    written from) that truth reference, so a mismatched reference is refused before anything runs. The tool is
    prepared once, then run per suite: on all reads (``run_suite``) or donor by donor (``run_genotype_arm``). Each
    run is recorded in its ``run.json``, failed ones included.
    """
    names = list(comma_list(args.suites))
    if not names:
        raise ValueError("--suites names no suite")
    if args.threads < 1:
        raise ValueError(f"--threads must be at least 1, got {args.threads}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be at least 1, got {args.batch_size}")
    base, label = parse_tool(args.tool)
    if label is not None:
        raise ValueError(f"tool {args.tool!r}: run takes plain tools; run {base} checkpoint variants with "
                         f"{VARIANT_COMMANDS.get(base, 'all')}")
    layout = RunLayout(absolute_path(args.run_dir))
    runner = make_runner(args.tool, RunnerOptions(
        igblast_dir=args.igblast_dir, threads=args.threads, cpus=args.cpus, alignair_image=args.image,
        alignair_model=args.model, alignair_checkpoint=args.checkpoint, device=args.device,
        batch_size=args.batch_size, cache_dir=args.cache_dir))
    reference = truth_reference(layout, names)
    if args.arm == "genotype":
        for name in names:
            if not layout.donors(name).is_file():
                raise ValueError(f"suite {name} has no donors ({layout.donors(name).name}); the genotype arm needs "
                                 "a suite simulated from donors")
    ensure_shared_reference(layout, reference)
    runner.prepare(layout, reference)
    failed = []
    for name in names:
        if args.arm == "genotype":
            result = run_genotype_arm(runner, layout, name, reference)
        else:
            result = run_suite(runner, layout, name, arm=args.arm)
        print(_summary_line(result))
        if result.returncode != 0:
            failed.append(result)
    for result in failed:
        donor = result.provenance.get("failed_donor")
        where = f" on donor {donor}" if donor else ""
        reason = f": {result.provenance['error']}" if result.provenance.get("error") else ""
        logs = (layout.donor_logs_dir(result.tool, result.suite, result.arm, donor) if donor
                else layout.logs_dir(result.tool, result.suite, result.arm))
        print(f"error: {result.tool} on {result.suite}/{result.arm} failed{where} with return code "
              f"{result.returncode}{reason}; see {logs}", file=sys.stderr)
    return 2 if failed else 0


def _summary_line(result: RunResult) -> str:
    rss = "not measured" if result.peak_rss_kb is None else f"{result.peak_rss_kb} KB"
    return (f"{result.tool} {result.suite}/{result.arm}: return code {result.returncode}, {result.n_reads} reads, "
            f"{result.wall_seconds:.1f} s, peak RSS {rss}; wrote {result.predictions}")
