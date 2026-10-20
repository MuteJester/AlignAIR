"""``diagnose``: AlignAIR's head metrics, stage contributions, calibration and speed profile on built dev suites, run
in process (``diagnostics.evaluate``) and cached per suite under the model key."""
from __future__ import annotations

import json
import sys

from alignair_assay.cli.common import comma_list, tool_path
from alignair_assay.core.layout import RunLayout
from alignair_assay.runners.base import absolute_path
from alignair_assay.workflow.build import build_changes

DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 256


def register(sub) -> None:
    """Add the ``diagnose`` subcommand."""
    p = sub.add_parser("diagnose", help="diagnose an AlignAIR checkpoint in process on built dev suites: head "
                                        "metrics, stage contributions, calibration, speed profile")
    p.add_argument("--run-dir", required=True, help="a run directory with the suites built on the dev split")
    p.add_argument("--checkpoint", required=True, help="AlignAIR .alignair model file")
    p.add_argument("--suites", required=True, help="comma-separated built suite names, e.g. S1,S3")
    p.add_argument("--device", default=DEFAULT_DEVICE, help=f"torch device (default {DEFAULT_DEVICE})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="reads per forward batch")
    p.add_argument("--variants", type=comma_list,
                   help="comma-separated truth-substitution variants (default: every one; the baseline is always "
                        "scored with another; an empty list runs only the head metrics, calibration and profile)")
    p.add_argument("--no-profile", action="store_true", help="skip the speed profile")
    p.add_argument("--max-cases", type=int, help="diagnose at most this many reads per suite, evenly spaced")
    p.set_defaults(func=run)


def run(args) -> int:
    """Diagnose the checkpoint on every named suite (``diagnostics.evaluate.evaluate``), print one line per suite
    and the profile, and name the report it wrote (``diagnostics.md`` and ``diagnostics.json`` under the model
    key). A suite whose build is not current for this checkout (``workflow.build.build_changes``) is diagnosed as
    built, with a warning. Input errors (an unbuilt or test-split suite, an unknown variant, a missing checkpoint,
    a model on another reference) exit with status 2 before anything is written."""
    # Imported here so the other commands never load torch and alignair.
    from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS
    from alignair_assay.diagnostics.evaluate import evaluate

    names = list(comma_list(args.suites))
    if not names:
        raise ValueError("--suites names no suite")
    layout = RunLayout(absolute_path(args.run_dir))
    _warn_stale_builds(layout, names)
    result = evaluate(tool_path(args.checkpoint), layout, names, device=args.device, batch_size=args.batch_size,
                      variants=DEFAULT_VARIANTS if args.variants is None else args.variants,
                      profile=not args.no_profile, max_cases=args.max_cases)
    for name, suite in result.suites.items():
        reads = suite["reads"]
        if suite["reused"]:
            print(f"{name}: reused the cached diagnostics of {reads['diagnosed']} reads (of {reads['built']}), "
                  f"computed in {suite['seconds']:.1f} s; {layout.diagnostics_suite_json(result.model_key, name)}")
        else:
            print(f"{name}: computed {reads['diagnosed']} reads (of {reads['built']}), {len(suite['variants'])} "
                  f"variants, in {suite['seconds']:.1f} s")
    if result.profile is not None:
        profile = result.profile
        print(f"profile: {profile['reads_per_second']:.1f} reads/s on {profile['device']} at batch size "
              f"{profile['batch_size']} over {profile['reads']} reads"
              + (" (reused)" if profile["reused"] else ""))
    print(f"report: {layout.diagnostics_md(result.model_key)}")
    return 0


def _warn_stale_builds(layout: RunLayout, names: list[str]) -> None:
    """Warn about each dev build whose identity is not this checkout's (``workflow.build.build_changes`` with its
    recorded configuration): its cases and truth come from other suite, truth or GenAIRR code. A suite that is not
    built, or not on the dev split, is left to ``evaluate``, which refuses it."""
    for name in names:
        try:
            build = json.loads(layout.build_json(name).read_text(encoding="utf-8"))
            if build.get("split") != "dev":
                continue
            changes = build_changes(layout, name, split=build["split"], reference_digest=build["reference_digest"],
                                    seed=build["seed"], scale=build["scale"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if changes:
            print(f"warning: {name}'s build is not current (changed: {', '.join(changes)}); it is diagnosed as "
                  f"built. Rebuild it to diagnose against this checkout's suite and truth code", file=sys.stderr)
