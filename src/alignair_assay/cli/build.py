"""``build``: generate suites into a run directory (cases, truth, reads and a build record per suite)."""
from __future__ import annotations

from alignair_assay.cli.common import comma_list
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.reference import DEFAULT_REFERENCE
from alignair_assay.suites import get_suite
from alignair_assay.suites.engine import DEFAULT_SEED, SPLITS, SuiteNotApplicable
from alignair_assay.workflow.build import build_suite


def register(sub) -> None:
    """Add the ``build`` subcommand."""
    p = sub.add_parser("build", help="generate suites into a run directory")
    p.add_argument("--suites", required=True, help="comma-separated suite names, e.g. S0,S1")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", choices=SPLITS, default="dev",
                   help="dev for model iteration (default); test is locked and used only for claims")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--scale", type=float, default=1.0, help="multiplies every stratum's case count")
    p.add_argument("--workers", type=int, default=8, help="processes generating strata in parallel")
    p.set_defaults(func=run)


def run(args) -> int:
    """Build every named suite (each name checked before anything is built) and print one line per suite; a suite
    that does not apply to the reference (``suites.engine.SuiteNotApplicable``) is reported and skipped."""
    names = list(comma_list(args.suites))
    if not names:
        raise ValueError("--suites names no suite")
    for name in names:
        get_suite(name)
    layout = RunLayout(args.run_dir)
    for name in names:
        try:
            build = build_suite(name, layout, reference_name=args.reference, split=args.split, seed=args.seed,
                                scale=args.scale, workers=args.workers)
        except SuiteNotApplicable as error:
            print(f"not applicable to {args.reference}: {name} ({error})")
            continue
        print(_summary_line(build, layout))
    return 0


def _summary_line(build: dict, layout: RunLayout) -> str:
    truth = build["truth"]
    strata = ", ".join(f"{name} {stats['cases']}" for name, stats in build["strata"].items())
    unscorable = " ".join(f"{g}={n}" for g, n in truth["unscorable"].items())
    return (f"{build['suite']}: {build['n_cases']} cases ({strata}); "
            f"excluded {sum(truth['excluded'].values())}, cases with warnings {len(truth['warnings'])}, "
            f"unscorable genes {unscorable}; {build['seconds']:.1f} s; wrote {layout.suite_dir(build['suite'])}")
