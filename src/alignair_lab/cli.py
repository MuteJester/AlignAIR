"""``python -m alignair_lab``: architecture experiments.

  python -m alignair_lab screen --name NAME [--steps 20000] [--architecture ARCH] [--model-option KEY=VALUE ...]
      [--schedule cosine|constant] [--init-seed 0] [--against-screen NAME | --against-milestone [DIR]]
      [--baseline-model-option KEY=VALUE ...] [--suites S1,S3] [--max-cases N] [--variants baseline,true_spans,...]
      [--device cuda] [--data-workers 2] [--eval-batch-size 256] [--cache DIR] [--dev-build DIR] [--out-root DIR]
  python -m alignair_lab cache [--out DIR] [--steps 40000] [--workers N] [--progress-seconds 60]
      [--segment-records N]

``screen`` trains an architecture with the recipe (``recipe``) from the record cache and compares it with a baseline
at the same step (``screen.run_screen``): a screen of lab-null-v1 is a baseline screen, which compares with nothing;
any other compares with the registered baseline screen of its protocol (or the one ``--against-screen`` names), or
with the baseline's milestone (``--against-milestone``, Tier 2; its directory defaults to the conv-v1 baseline's).
``cache`` writes the recipe's record cache (``records.write_record_cache``). A ``--model-option`` value is read as
JSON when it parses (``64``, ``true``,
``[3, 5]``), else as a string. Run under the frozen GenAIRR the cache was written with (``PYTHONPATH=FROZEN:src``).
A refused request or a failed run prints ``error: ...`` (and its hint) and returns 2.
"""
from __future__ import annotations

import argparse
import json
import sys

from alignair_lab import screen
from alignair_lab.recipe import BASELINE_ARCHITECTURE, BATCH_SIZE, DATACONFIGS, POLICY, SCHEDULES, SEED

CACHE_STEPS = 40_000


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        if getattr(error, "hint", None):
            print(f"hint: {error.hint}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m alignair_lab", description="AlignAIR architecture experiments.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("screen", help="train an architecture from the record cache and compare it with a "
                                             "baseline at the same step")
    run.add_argument("--name", required=True, help="the screen's name: its directory under --out-root")
    run.add_argument("--steps", type=int, default=screen.DEFAULT_STEPS, help="training steps (default %(default)s)")
    run.add_argument("--architecture", default=BASELINE_ARCHITECTURE,
                     help=f"a registered architecture, e.g. a candidate of alignair_lab.architectures; "
                          f"{screen.NULL_ARCHITECTURE} makes a baseline screen (default %(default)s)")
    run.add_argument("--model-option", action="append", default=[], metavar="KEY=VALUE",
                     help="a model configuration field (repeatable)")
    add_protocol_arguments(run)
    run.add_argument("--init-seed", type=int, default=screen.DEFAULT_INIT_SEED,
                     help="the seed of the model's initial weights, apart from the data seed (default %(default)s)")
    against = run.add_mutually_exclusive_group()
    against.add_argument("--against-screen", metavar="NAME",
                         help="the baseline screen to compare with (default: the registered baseline screen of the "
                              "protocol: lab-null-v1, init seed 0)")
    against.add_argument("--against-milestone", nargs="?", const=str(screen.MILESTONES), metavar="DIR",
                         help="compare with the baseline's milestone at the same step in DIR, model.step<N>.alignair "
                              f"(Tier 2; DIR defaults to {screen.MILESTONES}); its run kept a constant rate")
    run.add_argument("--baseline-model-option", action="append", default=[], metavar="KEY=VALUE",
                     help="a model configuration field of the baseline, when it is not its architecture's default "
                          "(repeatable; recorded in screen.json and on the headline)")
    run.add_argument("--variants", default=",".join(screen.DEFAULT_VARIANTS),
                     help="the assay's variants: baseline (head and end-to-end metrics, the default) and any "
                          "truth-substitution variant, e.g. baseline,true_spans")
    run.set_defaults(func=run_screen)

    cache = commands.add_parser("cache", help="write the recipe's record cache")
    cache.add_argument("--out", default=str(screen.CACHE), help="the new cache directory (default %(default)s)")
    cache.add_argument("--steps", type=int, default=CACHE_STEPS,
                       help=f"steps of {BATCH_SIZE} records the cache serves (default %(default)s)")
    cache.add_argument("--workers", type=int,
                       help="writer processes, at most one per segment written (the recipe's 12 selected streams are "
                            "cut into segments; its 16 unselected ones are written whole); default: one per physical "
                            "core, as GenAIRR gains nothing from a core's second hardware thread")
    cache.add_argument("--progress-seconds", type=float, default=60.,
                       help="seconds between progress lines (default %(default)s)")
    cache.add_argument("--segment-records", type=int,
                       help="write every stream in segments of this many records, each drawn from its own seed "
                            "(default: the selected streams only, sized to keep every writer busy; a size larger "
                            "than any stream writes each whole)")
    cache.set_defaults(func=run_cache)
    return parser


def add_protocol_arguments(command) -> None:
    """The arguments ``screen`` and ``calibrate`` share: the schedule, the assay's reads and the run's places."""
    command.add_argument("--schedule", choices=SCHEDULES, default=screen.DEFAULT_SCHEDULE,
                         help="the learning-rate schedule: cosine to a hundredth of the rate over the steps, or the "
                              "recipe's constant rate (default %(default)s)")
    command.add_argument("--suites", help="comma-separated dev suites (default: every suite built in --dev-build)")
    command.add_argument("--max-cases", type=int, help="reads per suite, evenly spaced (default: all)")
    command.add_argument("--device", default="cuda")
    command.add_argument("--data-workers", type=int, default=screen.DEFAULT_DATA_WORKERS,
                         help="training data worker processes (default %(default)s)")
    command.add_argument("--eval-batch-size", type=int, default=256,
                         help="the assay's batch size (default %(default)s)")
    command.add_argument("--cache", default=str(screen.CACHE), help="the record cache (default %(default)s)")
    command.add_argument("--dev-build", default=str(screen.DEV_BUILD),
                         help="the assay's dev build (default %(default)s)")
    command.add_argument("--out-root", default=str(screen.SCREENS),
                         help="where screens are written (default %(default)s)")


def model_options(pairs) -> dict:
    """``KEY=VALUE`` pairs as model options, each value read as JSON when it parses, else as a string."""
    options = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"--model-option takes KEY=VALUE, got {pair!r}")
        try:
            options[key] = json.loads(value)
        except ValueError:
            options[key] = value
    return options


def _names(text: str | None) -> list[str] | None:
    return None if text is None else [name.strip() for name in text.split(",") if name.strip()]


def run_screen(args) -> int:
    screen.run_screen(args.name, steps=args.steps, architecture=args.architecture,
                      model_options=model_options(args.model_option), suites=_names(args.suites),
                      max_cases=args.max_cases, variants=_names(args.variants), device=args.device,
                      data_workers=args.data_workers, eval_batch_size=args.eval_batch_size, cache=args.cache,
                      dev_build=args.dev_build, out_root=args.out_root, schedule=args.schedule,
                      init_seed=args.init_seed, against_screen=args.against_screen,
                      against_milestone=args.against_milestone,
                      baseline_options=model_options(args.baseline_model_option))
    return 0


def run_cache(args) -> int:
    from alignair_lab.records import write_record_cache
    manifest = write_record_cache(args.out, dataconfigs=DATACONFIGS, policy=POLICY, seed=SEED, steps=args.steps,
                                  batch_size=BATCH_SIZE, workers=args.workers, segment_records=args.segment_records,
                                  progress=lambda line: print(f"[cache] {line}", flush=True),
                                  progress_seconds=args.progress_seconds)
    print(f"wrote {args.out}: {manifest['records']} records ({manifest['steps']} steps of {manifest['batch_size']}) "
          f"in {len(manifest['components'])} streams, {manifest['bytes'] / 1e9:.2f} GB, in "
          f"{manifest['seconds'] / 60:.1f} min with {manifest['workers']} workers")
    return 0
