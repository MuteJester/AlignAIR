"""CLI handlers for benchmark contract validation commands."""
from __future__ import annotations

import json

from ..core import validate_artifact, validate_catalogs
from ..evaluation import validate_benchmark_report_contract, validate_comparison_policy_catalog
from .common import emit_json
from .options import ARTIFACT_KIND_CHOICES


def register_validation_commands(subparsers) -> None:
    """Register benchmark contract validation commands."""

    validate_catalogs_parser = subparsers.add_parser("validate-catalogs", help="validate criteria and scenario catalogs")
    validate_catalogs_parser.add_argument("--out", default=None, help="optional output validation JSON path")
    validate_catalogs_parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="exit nonzero when validation emits warnings",
    )
    validate_catalogs_parser.set_defaults(func=validate_catalogs_command)

    validate_comparison_policies = subparsers.add_parser(
        "validate-comparison-policies",
        help="validate built-in comparison policy metrics against the metric registry",
    )
    validate_comparison_policies.add_argument("--out", default=None, help="optional output validation JSON path")
    validate_comparison_policies.set_defaults(func=validate_comparison_policies_command)

    validate_report = subparsers.add_parser("validate-report", help="validate benchmark report artifact and metric contracts")
    validate_report.add_argument("--report", required=True, help="benchmark report JSON")
    validate_report.add_argument("--out", default=None, help="optional output validation JSON path")
    validate_report.add_argument(
        "--require-current-version",
        action="store_true",
        help="fail if embedded schema or metric registry differs from the current code",
    )
    validate_report.add_argument(
        "--require-metric-registry",
        action="store_true",
        help="fail legacy reports that do not embed metric_registry",
    )
    validate_report.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="exit nonzero when validation emits warnings",
    )
    validate_report.set_defaults(func=validate_report_command)

    validate_artifact_parser = subparsers.add_parser("validate-artifact", help="validate a registered benchmark artifact JSON")
    validate_artifact_parser.add_argument("--path", required=True, help="artifact JSON path")
    validate_artifact_parser.add_argument("--kind", required=True, choices=ARTIFACT_KIND_CHOICES, help="artifact kind")
    validate_artifact_parser.add_argument("--out", default=None, help="optional output validation JSON path")
    validate_artifact_parser.add_argument(
        "--require-current-version",
        action="store_true",
        help="fail if embedded schema metadata is not the current version",
    )
    validate_artifact_parser.set_defaults(func=validate_artifact_command)


def validate_catalogs_command(args) -> None:
    validation = validate_catalogs()
    emit_json(validation, args.out)
    if not validation["valid"] or (args.fail_on_warning and validation["warnings"]):
        raise SystemExit(1)


def validate_comparison_policies_command(args) -> None:
    validation = validate_comparison_policy_catalog()
    emit_json(validation, args.out)
    if not validation["valid"]:
        raise SystemExit(1)


def validate_report_command(args) -> None:
    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    validation = validate_benchmark_report_contract(
        report,
        require_current_version=args.require_current_version,
        require_metric_registry=args.require_metric_registry,
    )
    emit_json(validation, args.out)
    if not validation["valid"] or (args.fail_on_warning and validation["warnings"]):
        raise SystemExit(1)


def validate_artifact_command(args) -> None:
    with open(args.path, encoding="utf-8") as handle:
        payload = json.load(handle)
    validation = validate_artifact(
        payload,
        args.kind,
        require_current_version=args.require_current_version,
    )
    emit_json(validation, args.out)
    if not validation["valid"]:
        raise SystemExit(1)
