"""
CLI entry point for AlignAIR Benchmarking.

Usage:
    python -m AlignAIR.Benchmarking snapshot --model-dir PATH --eval-data PATH --output-dir PATH
    python -m AlignAIR.Benchmarking compare --baseline PATH --current PATH [--tolerance-profile PROFILE]
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cmd_snapshot(args):
    from AlignAIR.Benchmarking.snapshot import ModelSnapshot

    output = ModelSnapshot.create(
        model_dir=args.model_dir,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_sequences=args.max_sequences,
        include_latent=not args.no_latent,
        include_pipeline=not args.no_pipeline,
    )
    print(f"Snapshot created at: {output}")


def cmd_compare(args):
    from AlignAIR.Benchmarking.compare import SnapshotComparator
    from AlignAIR.Benchmarking.report import text_report, save_report

    comparator = SnapshotComparator.from_dirs(
        baseline_dir=args.baseline,
        current_dir=args.current,
        tolerance_profile=args.tolerance_profile,
    )
    result = comparator.compare_all()

    # Print to console
    print(text_report(result))

    # Optionally save to file
    if args.output:
        fmt = "json" if args.output.endswith(".json") else "text"
        save_report(result, args.output, format=fmt)
        print(f"\nReport saved to: {args.output}")

    # Exit code: 0 for PASS, 1 for FAIL
    if result.overall_status == "FAIL":
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="AlignAIR.Benchmarking",
        description="AlignAIR reproducibility and regression testing framework.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- snapshot --
    snap = subparsers.add_parser("snapshot", help="Create a reproducibility snapshot")
    snap.add_argument("--model-dir", required=True, help="Path to model bundle directory")
    snap.add_argument("--eval-data", required=True, help="Path to evaluation CSV")
    snap.add_argument("--output-dir", required=True, help="Where to save the snapshot")
    snap.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    snap.add_argument("--max-sequences", type=int, default=None, help="Limit number of sequences")
    snap.add_argument("--no-latent", action="store_true", help="Skip latent representation extraction")
    snap.add_argument("--no-pipeline", action="store_true", help="Skip full pipeline execution")

    # -- compare --
    cmp = subparsers.add_parser("compare", help="Compare two snapshots")
    cmp.add_argument("--baseline", required=True, help="Path to baseline snapshot directory")
    cmp.add_argument("--current", required=True, help="Path to current snapshot directory")
    cmp.add_argument("--tolerance-profile", default="default",
                     choices=["code-change", "default", "model-comparison"],
                     help="Tolerance profile to use")
    cmp.add_argument("--output", default=None, help="Save report to file (.txt or .json)")

    args = parser.parse_args()

    if args.command == "snapshot":
        cmd_snapshot(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
