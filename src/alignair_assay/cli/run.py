"""``run``: run one tool on built suites of a run directory, against the shared reference."""
from __future__ import annotations

import sys

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.reference import Reference
from alignair_assay.runners import RUNNERS, make_runner
from alignair_assay.runners.base import RunnerOptions, RunResult, run_suite
from alignair_assay.runners.reference import ensure_shared_reference
from alignair_assay.truth.io import read_truth_header

ARMS = ("shared",)


def register(sub) -> None:
    """Add the ``run`` subcommand."""
    p = sub.add_parser("run", help="run one tool on built suites of a run directory")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tool", required=True, help=f"registered tool name ({', '.join(RUNNERS)})")
    p.add_argument("--suites", required=True, help="comma-separated built suite names, e.g. S0,S1")
    p.add_argument("--arm", choices=ARMS, default="shared", help="shared: every tool gets the shared reference")
    p.add_argument("--igblast-dir", help="directory of an unpacked IgBLAST release (igblast)")
    p.add_argument("--threads", type=int, default=8, help="tool threads")
    p.add_argument("--cpus", help="CPU list to pin the tool to with taskset, e.g. 0-7")
    p.set_defaults(func=run)


def run(args) -> int:
    """Run the tool on every named suite and print one line per run; 2 when any run failed.

    Every input is checked first: the tool name and options, and that each suite is built against one truth
    reference. The shared reference is then verified against (or written from) that truth reference, so a
    mismatched reference is refused before anything runs; the tool is prepared once, then run per suite. Each
    run is recorded in its ``run.json``, failed ones included.
    """
    names = list(dict.fromkeys(name.strip() for name in args.suites.split(",") if name.strip()))
    if not names:
        raise ValueError("--suites names no suite")
    if args.threads < 1:
        raise ValueError(f"--threads must be at least 1, got {args.threads}")
    layout = RunLayout(args.run_dir)
    runner = make_runner(args.tool, RunnerOptions(igblast_dir=args.igblast_dir, threads=args.threads,
                                                  cpus=args.cpus))
    reference = _truth_reference(layout, names)
    ensure_shared_reference(layout, reference)
    runner.prepare(layout, reference)
    failed = []
    for name in names:
        result = run_suite(runner, layout, name, arm=args.arm)
        print(_summary_line(result))
        if result.returncode != 0:
            failed.append(result)
    for result in failed:
        print(f"error: {result.tool} on {result.suite}/{result.arm} failed with return code {result.returncode}; "
              f"see {layout.logs_dir(result.tool, result.suite, result.arm)}", file=sys.stderr)
    return 2 if failed else 0


def _truth_reference(layout: RunLayout, suites: list[str]) -> Reference:
    """The reference the named suites' truth was built against, from its truth headers. ValueError when a suite
    is not built (no ``build.json``), the suites name different references, or the reference's dataconfig no
    longer has the recorded digest."""
    recorded = {}
    for suite in suites:
        if not layout.build_json(suite).is_file():
            raise ValueError(f"suite {suite} is not built in {layout.root} (no {layout.build_json(suite).name}); "
                             f"build it first")
        header = read_truth_header(layout.truth(suite)) or {}
        recorded[suite] = (header.get("reference"), header.get("reference_digest"))
    if len(set(recorded.values())) > 1:
        listed = ", ".join(f"{suite}: {name} ({digest and digest[:12]})" for suite, (name, digest) in recorded.items())
        raise ValueError(f"the suites were built against different references: {listed}")
    name, digest = next(iter(recorded.values()))
    if name is None or digest is None:
        raise ValueError(f"{layout.truth(suites[0])} records no reference digest; rebuild the suite")
    reference = Reference.from_dataconfig(name)
    if reference.digest() != digest:
        raise ValueError(f"the truth was built against {name} with digest {digest[:12]}, but {name} now has digest "
                         f"{reference.digest()[:12]}; rebuild the suites")
    return reference


def _summary_line(result: RunResult) -> str:
    rss = "not measured" if result.peak_rss_kb is None else f"{result.peak_rss_kb} KB"
    return (f"{result.tool} {result.suite}/{result.arm}: return code {result.returncode}, {result.n_reads} reads, "
            f"{result.wall_seconds:.1f} s, peak RSS {rss}; wrote {result.predictions}")
