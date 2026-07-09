"""CLI handlers for benchmark IO and readiness commands."""
from __future__ import annotations

import json

from ..generation import assess_benchmark_readiness, coverage_summary, dataconfig_by_name
from ..io import export_benchmark_inputs, load_jsonl
from alignair.reference.reference_set import ReferenceSet
from .common import emit_json
from .options import READINESS_PROFILE_CHOICES


def register_io_commands(subparsers) -> None:
    """Register benchmark IO and readiness commands."""

    summary_parser = subparsers.add_parser("summary", help="print coverage summary for a benchmark JSONL")
    summary_parser.add_argument("path")
    summary_parser.set_defaults(func=summary)

    export_parser = subparsers.add_parser("export", help="export benchmark cases for external alignment tools")
    export_parser.add_argument("--cases", required=True, help="benchmark case JSONL")
    export_parser.add_argument("--out-dir", required=True, help="directory for FASTA/AIRR/manifest files")
    export_parser.add_argument("--prefix", default="benchmark", help="output filename prefix")
    export_parser.add_argument(
        "--frame",
        choices=("presented", "canonical"),
        default="presented",
        help="sequence frame for exported input sequences",
    )
    export_parser.add_argument("--config", default=None, help="optional GenAIRR DataConfig name for reference summary")
    export_parser.add_argument("--airr-metadata", action="store_true", help="include benchmark metadata in AIRR input TSV")
    export_parser.add_argument(
        "--readiness-profile",
        choices=READINESS_PROFILE_CHOICES,
        default="assay",
        help="readiness profile written into exported manifest",
    )
    export_parser.set_defaults(func=export)

    readiness_parser = subparsers.add_parser("readiness", help="assess generated benchmark coverage before model evaluation")
    readiness_parser.add_argument("--cases", required=True, help="benchmark case JSONL")
    readiness_parser.add_argument("--config", default=None, help="optional GenAIRR DataConfig name for reference coverage")
    readiness_parser.add_argument(
        "--profile",
        choices=READINESS_PROFILE_CHOICES,
        default="assay",
        help="readiness threshold profile",
    )
    readiness_parser.add_argument("--out", default=None, help="optional output readiness JSON path")
    readiness_parser.set_defaults(func=readiness)


def summary(args) -> None:
    cases = load_jsonl(args.path)
    print(json.dumps(coverage_summary(cases), indent=2, sort_keys=True))


def export(args) -> None:
    cases = load_jsonl(args.cases)
    reference_set = None
    if args.config:
        dataconfig = dataconfig_by_name(args.config)
        reference_set = ReferenceSet.from_dataconfigs(dataconfig)
    files = export_benchmark_inputs(
        cases,
        args.out_dir,
        prefix=args.prefix,
        frame=args.frame,
        include_airr_metadata=args.airr_metadata,
        dataconfig_name=args.config,
        reference_set=reference_set,
        readiness_profile=args.readiness_profile,
    )
    print(json.dumps({"n_cases": len(cases), "exports": files}, indent=2, sort_keys=True))


def readiness(args) -> None:
    cases = load_jsonl(args.cases)
    reference_set = None
    if args.config:
        dataconfig = dataconfig_by_name(args.config)
        reference_set = ReferenceSet.from_dataconfigs(dataconfig)
    report = assess_benchmark_readiness(
        cases,
        reference_set=reference_set,
        profile=args.profile,
    )
    emit_json(report, args.out)
