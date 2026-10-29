"""``python -m alignair_lab``: architecture experiments.

  python -m alignair_lab screen --name NAME [--steps 20000] [--architecture ARCH] [--model-option KEY=VALUE ...]
      [--suites S1,S3] [--max-cases N] [--variants baseline,true_spans,...] [--device cuda] [--data-workers 2]
      [--eval-batch-size 256] [--cache DIR] [--dev-build DIR] [--milestones DIR] [--out-root DIR]
  python -m alignair_lab cache [--out DIR] [--steps 40000] [--workers N]

``screen`` trains a candidate with the 1.0.0 recipe from the record cache and compares it with the baseline's
milestone at the same step (``screen.run_screen``); ``cache`` writes the recipe's record cache
(``records.write_record_cache``). A ``--model-option`` value is read as JSON when it parses (``64``, ``true``,
``[3, 5]``), else as a string. Run under the frozen GenAIRR the cache was written with (``PYTHONPATH=FROZEN:src``).
A refused request or a failed run prints ``error: ...`` (and its hint) and returns 2.
"""
from __future__ import annotations

import argparse
import json
import sys

from alignair_lab import screen
from alignair_lab.recipe import BASELINE_ARCHITECTURE, BATCH_SIZE, DATACONFIGS, POLICY, SEED

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

    run = commands.add_parser("screen", help="train a candidate from the record cache and compare it with the "
                                             "baseline at the same step")
    run.add_argument("--name", required=True, help="the screen's name: its directory under --out-root")
    run.add_argument("--steps", type=int, default=screen.DEFAULT_STEPS,
                     help="training steps; the baseline must have a milestone at this step (default %(default)s)")
    run.add_argument("--architecture", default=BASELINE_ARCHITECTURE,
                     help="a registered architecture, e.g. a candidate of alignair_lab.architectures "
                          "(default %(default)s)")
    run.add_argument("--model-option", action="append", default=[], metavar="KEY=VALUE",
                     help="a model configuration field (repeatable)")
    run.add_argument("--suites", help="comma-separated dev suites (default: every suite built in --dev-build)")
    run.add_argument("--max-cases", type=int, help="reads per suite, evenly spaced (default: all)")
    run.add_argument("--variants", default=",".join(screen.DEFAULT_VARIANTS),
                     help="the assay's variants: baseline (head and end-to-end metrics, the default) and any "
                          "truth-substitution variant, e.g. baseline,true_spans")
    run.add_argument("--device", default="cuda")
    run.add_argument("--data-workers", type=int, default=screen.DEFAULT_DATA_WORKERS,
                     help="training data worker processes (default %(default)s)")
    run.add_argument("--eval-batch-size", type=int, default=256, help="the assay's batch size (default %(default)s)")
    run.add_argument("--cache", default=str(screen.CACHE), help="the record cache (default %(default)s)")
    run.add_argument("--dev-build", default=str(screen.DEV_BUILD), help="the assay's dev build (default %(default)s)")
    run.add_argument("--milestones", default=str(screen.MILESTONES),
                     help="the baseline's milestone checkpoints, model.step<N>.alignair (default %(default)s)")
    run.add_argument("--out-root", default=str(screen.SCREENS), help="where screens are written (default %(default)s)")
    run.set_defaults(func=run_screen)

    cache = commands.add_parser("cache", help="write the 1.0.0 recipe's record cache")
    cache.add_argument("--out", default=str(screen.CACHE), help="the new cache directory (default %(default)s)")
    cache.add_argument("--steps", type=int, default=CACHE_STEPS,
                       help=f"steps of {BATCH_SIZE} records the cache serves (default %(default)s)")
    cache.add_argument("--workers", type=int, help="writer processes (default: one per curriculum component)")
    cache.set_defaults(func=run_cache)
    return parser


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
                      dev_build=args.dev_build, milestones=args.milestones, out_root=args.out_root)
    return 0


def run_cache(args) -> int:
    from alignair_lab.records import write_record_cache
    manifest = write_record_cache(args.out, dataconfigs=DATACONFIGS, policy=POLICY, seed=SEED, steps=args.steps,
                                  batch_size=BATCH_SIZE, workers=args.workers)
    print(f"wrote {args.out}: {manifest['records']} records ({manifest['steps']} steps of {manifest['batch_size']}) "
          f"in {len(manifest['components'])} components, {manifest['bytes'] / 1e9:.2f} GB, in "
          f"{manifest['seconds'] / 60:.1f} min with {manifest['workers']} workers")
    return 0
