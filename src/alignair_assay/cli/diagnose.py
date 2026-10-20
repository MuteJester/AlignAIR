"""``diagnose``: AlignAIR's head metrics, stage contributions, calibration and speed profile on built dev suites, run
in process (``workflow.diagnose``) and cached per suite under the model key."""
from __future__ import annotations

import sys

from alignair_assay.cli.common import comma_list, tool_path
from alignair_assay.core.layout import RunLayout
from alignair_assay.runners.base import absolute_path

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
    p.add_argument("--allow-unverified-replay", action="store_true",
                   help="diagnose even when the AlignAIR code the replay mirrors has changed since it was pinned "
                        "(the results are marked unverified)")
    p.set_defaults(func=run)


def run(args) -> int:
    """Diagnose the checkpoint on every named suite (``workflow.diagnose.evaluate``), print one line per suite
    and the profile, and name the report it wrote (the one-page ``diagnostics.md``, with ``diagnostics.json`` and
    each suite's ``<S>/report.md``, under the model key). A suite whose build is not current for this checkout
    (``workflow.diagnose.stale_builds``) is diagnosed as built, with a warning. Input errors (an unbuilt or
    test-split suite, an unknown variant, a missing checkpoint, a model on another reference, a CUDA device torch
    does not see) exit with status 2 before anything is written,
    and so does AlignAIR code the replay is not pinned to, unless ``--allow-unverified-replay`` (then the results
    are marked unverified, with a warning)."""
    # Imported here so the other commands never load torch and alignair.
    from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS
    from alignair_assay.workflow.diagnose import evaluate, parse_device

    names = list(comma_list(args.suites))
    if not names:
        raise ValueError("--suites names no suite")
    device = parse_device(args.device, fix="--device cpu")
    layout = RunLayout(absolute_path(args.run_dir))
    result = evaluate(tool_path(args.checkpoint), layout, names, device=device, batch_size=args.batch_size,
                      variants=DEFAULT_VARIANTS if args.variants is None else args.variants,
                      profile=not args.no_profile, max_cases=args.max_cases,
                      allow_unverified_replay=args.allow_unverified_replay)
    if not result.replay_verified:
        print(f"warning: diagnosed with an unverified replay (changed: {', '.join(result.replay_changes)}); the "
              "results are marked unverified", file=sys.stderr)
    for name, changes in result.stale_builds.items():
        print(f"warning: {name}'s build is not current (changed: {', '.join(changes)}); it was diagnosed as "
              f"built. Rebuild it to diagnose against this checkout's suite and truth code", file=sys.stderr)
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
    print(f"report: {layout.diagnostics_md(result.model_key)} (one page; each suite's detail in "
          f"{layout.diagnostics_dir(result.model_key)}/<S>/report.md)")
    return 0
