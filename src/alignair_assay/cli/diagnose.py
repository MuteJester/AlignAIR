"""``diagnose``: AlignAIR's head metrics, stage contributions, calibration and speed profile on built dev suites, run
in process (``workflow.diagnose``) and cached per suite under the model key."""
from __future__ import annotations

import sys
from pathlib import Path

from alignair_assay.cli.common import comma_list, labelled_path, tool_path
from alignair_assay.core.layout import RunLayout
from alignair_assay.report.diagnostics_against import (
    AGAINST_FAMILIES,
    AGAINST_HEADLINE,
    AGAINST_VERDICTS,
    FAMILY_LABELS,
    code_text,
)
from alignair_assay.report.markdown import format_value
from alignair_assay.runners.base import absolute_path

DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 256


def register(sub) -> None:
    """Add the ``diagnose`` subcommand."""
    p = sub.add_parser("diagnose", help="diagnose an AlignAIR checkpoint in process on built dev suites: head "
                                        "metrics, stage contributions, calibration, speed profile")
    p.add_argument("--run-dir", required=True, help="a run directory with the suites built on the dev split")
    p.add_argument("--checkpoint", required=True, type=labelled_path, metavar="[LABEL=]PATH",
                   help="AlignAIR .alignair model file, or LABEL=PATH to name it LABEL in the report")
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
    p.add_argument("--against", metavar="PATH_OR_MODEL_KEY",
                   help="compare with a baseline model on the same reads: a .alignair checkpoint (diagnosed first "
                        "with these settings) or the model key of one already diagnosed in --run-dir. Δ = "
                        "--checkpoint − the baseline; fixed = reads the baseline gets wrong and --checkpoint right, "
                        "broken = the baseline right and --checkpoint wrong")
    p.set_defaults(func=run)


def run(args) -> int:
    """Diagnose the checkpoint on every named suite (``workflow.diagnose.evaluate``), print one line per suite
    and the profile, and name the report it wrote (the one-page ``diagnostics.md``, with ``diagnostics.json`` and
    each suite's ``<S>/report.md``, under the model key). A suite whose build is not current for this checkout
    (``workflow.diagnose.stale_builds``) is diagnosed as built, with a warning. Input errors (an unbuilt or
    test-split suite, an unknown variant, a missing checkpoint, a model on another reference, a CUDA device torch
    does not see) exit with status 2 before anything is written,
    and so does AlignAIR code the replay is not pinned to, unless ``--allow-unverified-replay`` (then the results
    are marked unverified, with a warning).

    With ``--against`` (``evaluate``'s ``against``, the baseline): an existing file is a checkpoint (made absolute),
    else a model key; a missing checkpoint, a model key without current diagnostics of every suite, or diagnostics that
    differ in anything but the model exit with status 2. It then prints the comparison: per suite the headline deltas
    (this model − the baseline) and the reads fixed (the baseline wrong, this model right) and broken, and the detail
    file."""
    # Imported here so the other commands never load torch and alignair.
    from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS
    from alignair_assay.workflow.diagnose import MODEL_KEY, evaluate, parse_device

    names = list(comma_list(args.suites))
    if not names:
        raise ValueError("--suites names no suite")
    device = parse_device(args.device, fix="--device cpu")
    layout = RunLayout(absolute_path(args.run_dir))
    against = args.against
    if against is not None and (Path(against).expanduser().exists() or not MODEL_KEY.fullmatch(against)):
        against = tool_path(against)
    label, checkpoint = args.checkpoint
    result = evaluate(checkpoint, layout, names, device=device, batch_size=args.batch_size, label=label,
                      variants=DEFAULT_VARIANTS if args.variants is None else args.variants,
                      profile=not args.no_profile, max_cases=args.max_cases,
                      allow_unverified_replay=args.allow_unverified_replay, against=against)
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
    if result.against is not None:
        _print_against(result, layout)
    print(f"report: {layout.diagnostics_md(result.model_key)} (one page; each suite's detail in "
          f"{layout.diagnostics_dir(result.model_key)}/<S>/report.md)")
    return 0


def _print_against(result, layout: RunLayout) -> None:
    """The comparison with the baseline: which model and the AlignAIR code each ran under (flagging a different
    code), then per suite the headline deltas (this − baseline), the
    significant metrics with their direction (this better, baseline better), the reads fixed and broken and the
    detail file."""
    summary = result.against["summary"]
    other = summary["baseline"]
    same = " (the same model key: one record compared with itself)" if summary["same_model"] else ""
    print(f"against the baseline {other} ({summary['source'] or 'source not recorded'}){same}: Δ = this model − the "
          "baseline; fixed = reads the baseline gets wrong and this model right, broken = the baseline right and "
          "this model wrong")
    print(code_text(summary["code"]).replace("**", ""))
    for name, suite in summary["suites"].items():
        deltas = ", ".join(f"{metric} {format_value(metric, entry['delta'], delta=True)}"
                           for metric in AGAINST_HEADLINE if (entry := suite["headline"].get(metric)))
        reads = ", ".join(f"{FAMILY_LABELS[family]} fixed {counts['fixed']}, broken {counts['broken']}"
                          for family, counts in suite["reads"].items())
        verdicts = {metric: entry.get("verdict") for family in AGAINST_FAMILIES
                    for metric, entry in ((result.against["suites"][name].get(family) or {}).get("comparison")
                                          or {}).items()}
        significant = ", ".join(f"{metric} ({AGAINST_VERDICTS.get(verdicts.get(metric), verdicts.get(metric))})"
                                for metric in suite["significant"]) or "none"
        print(f"{name} against {other}: {deltas or 'no end-to-end metric'}; significant: {significant}; reads "
              f"{reads}; {layout.diagnostics_against_md(result.model_key, name, other)}")
