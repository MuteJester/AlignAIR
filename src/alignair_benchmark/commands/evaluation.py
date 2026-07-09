"""CLI handlers for benchmark evaluation and reporting commands."""
from __future__ import annotations

import json

from ..evaluation import (
    audit_criteria_report,
    build_assay_report,
    build_benchmark_report,
    build_model_comparison_report,
    normalize_performance_summary,
)
from ..io import (
    load_airr_predictions,
    load_dicts_jsonl,
    load_jsonl,
    save_dicts_jsonl,
)
from .common import emit_json
from .options import COMPARISON_POLICY_CHOICES, MULTIPLE_COMPARISON_CORRECTIONS, PREDICTION_FORMATS


def register_evaluation_commands(subparsers) -> None:
    """Register benchmark scoring and model-comparison commands."""

    assay_parser = subparsers.add_parser("assay", help="build an assay-style report from score/report JSON")
    assay_parser.add_argument("path")
    assay_parser.add_argument("--top-contexts", type=int, default=25)
    assay_parser.set_defaults(func=assay)

    normalize_parser = subparsers.add_parser(
        "normalize-predictions",
        help="normalize prediction files to benchmark JSONL",
    )
    normalize_parser.add_argument("--input", required=True, help="prediction file to normalize")
    normalize_parser.add_argument(
        "--format",
        choices=PREDICTION_FORMATS,
        default="airr",
        help="input prediction format",
    )
    normalize_parser.add_argument("--delimiter", default=None, help="delimiter override for AIRR tables, e.g. '\\t'")
    normalize_parser.add_argument("--out", required=True, help="normalized prediction JSONL path")
    normalize_parser.set_defaults(func=normalize_predictions)

    evaluate_parser = subparsers.add_parser("evaluate", help="score prediction files against benchmark cases")
    evaluate_parser.add_argument("--cases", required=True, help="benchmark case JSONL")
    evaluate_parser.add_argument("--predictions", required=True, help="prediction file")
    evaluate_parser.add_argument(
        "--prediction-format",
        choices=PREDICTION_FORMATS,
        default="jsonl",
        help="prediction file format",
    )
    evaluate_parser.add_argument("--delimiter", default=None, help="delimiter override for AIRR tables, e.g. '\\t'")
    evaluate_parser.add_argument("--out", default=None, help="optional output report JSON path")
    evaluate_parser.add_argument(
        "--performance-json",
        default=None,
        help="optional JSON sidecar with runtime/memory stats for externally generated predictions",
    )
    evaluate_parser.add_argument("--frame", choices=("canonical", "presented"), default="canonical")
    evaluate_parser.add_argument("--contract-level", choices=("minimal", "core", "assay"), default=None)
    evaluate_parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="number of paired case-bootstrap replicates for confidence intervals; 0 disables",
    )
    evaluate_parser.add_argument("--confidence", type=float, default=0.95, help="bootstrap confidence level")
    evaluate_parser.add_argument("--bootstrap-seed", type=int, default=123, help="bootstrap RNG seed")
    evaluate_parser.add_argument(
        "--no-bootstrap-strata",
        dest="bootstrap_strata",
        action="store_false",
        help="skip per-stratum bootstrap intervals while keeping overall bootstrap intervals",
    )
    evaluate_parser.set_defaults(bootstrap_strata=True)
    evaluate_parser.add_argument(
        "--match-by",
        default="sequence_id",
        help="prediction field used to align rows to cases; use 'order' for list-order scoring",
    )
    evaluate_parser.add_argument(
        "--duplicate-policy",
        choices=("first", "last", "error"),
        default="first",
        help="how to handle duplicate prediction ids when --match-by is not 'order'",
    )
    evaluate_parser.set_defaults(func=evaluate)

    compare_parser = subparsers.add_parser("compare", help="compare two prediction files on paired benchmark cases")
    compare_parser.add_argument("--cases", required=True, help="benchmark case JSONL")
    compare_parser.add_argument("--a-predictions", required=True, help="model A prediction file")
    compare_parser.add_argument("--b-predictions", required=True, help="model B prediction file")
    compare_parser.add_argument("--model-a-name", default="model_a", help="label for model A")
    compare_parser.add_argument("--model-b-name", default="model_b", help="label for model B")
    compare_parser.add_argument(
        "--prediction-format",
        choices=PREDICTION_FORMATS,
        default="jsonl",
        help="prediction file format used for both models unless overridden",
    )
    compare_parser.add_argument(
        "--a-prediction-format",
        choices=PREDICTION_FORMATS,
        default=None,
        help="optional model A prediction format override",
    )
    compare_parser.add_argument(
        "--b-prediction-format",
        choices=PREDICTION_FORMATS,
        default=None,
        help="optional model B prediction format override",
    )
    compare_parser.add_argument("--delimiter", default=None, help="delimiter override used for both AIRR tables")
    compare_parser.add_argument("--a-delimiter", default=None, help="optional model A delimiter override")
    compare_parser.add_argument("--b-delimiter", default=None, help="optional model B delimiter override")
    compare_parser.add_argument("--out", default=None, help="optional output comparison JSON path")
    compare_parser.add_argument("--frame", choices=("canonical", "presented"), default="canonical")
    compare_parser.add_argument(
        "--policy",
        choices=COMPARISON_POLICY_CHOICES,
        default=None,
        help="built-in endpoint/guardrail policy template",
    )
    compare_parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="metric path to compare; repeatable; defaults to the standard comparison metrics",
    )
    compare_parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="number of paired case-bootstrap replicates for model-difference intervals; 0 disables",
    )
    compare_parser.add_argument("--confidence", type=float, default=0.95, help="bootstrap confidence level")
    compare_parser.add_argument("--bootstrap-seed", type=int, default=123, help="bootstrap RNG seed")
    compare_parser.add_argument(
        "--multiple-comparison-correction",
        choices=MULTIPLE_COMPARISON_CORRECTIONS,
        default="none",
        help="familywise correction used by endpoint/guardrail decision gates",
    )
    compare_parser.add_argument(
        "--no-bootstrap-strata",
        dest="bootstrap_strata",
        action="store_false",
        help="skip per-stratum comparison sections while keeping the overall paired comparison",
    )
    compare_parser.set_defaults(bootstrap_strata=True)
    compare_parser.add_argument(
        "--practical-delta",
        type=float,
        default=0.0,
        help="minimum direction-adjusted aggregate delta needed for a better/worse verdict",
    )
    compare_parser.add_argument(
        "--case-tie-tolerance",
        type=float,
        default=0.0,
        help="per-case direction-adjusted delta treated as a win instead of a tie",
    )
    compare_parser.add_argument(
        "--primary-metric",
        action="append",
        default=None,
        help="primary endpoint metric path for the policy decision; repeatable",
    )
    compare_parser.add_argument(
        "--guardrail-metric",
        action="append",
        default=None,
        help="no-regression guardrail metric path for the policy decision; repeatable",
    )
    compare_parser.add_argument(
        "--minimum-primary-advantage",
        type=float,
        default=None,
        help="minimum direction-adjusted primary endpoint advantage required to pass the gate",
    )
    compare_parser.add_argument(
        "--maximum-guardrail-regression",
        type=float,
        default=None,
        help="maximum allowed direction-adjusted regression on each guardrail metric",
    )
    compare_parser.add_argument(
        "--match-by",
        default="sequence_id",
        help="prediction field used to align both model outputs to cases; use 'order' for list-order scoring",
    )
    compare_parser.add_argument(
        "--duplicate-policy",
        choices=("first", "last", "error"),
        default="first",
        help="how to handle duplicate prediction ids when --match-by is not 'order'",
    )
    compare_parser.set_defaults(func=compare)

    audit_parser = subparsers.add_parser("audit", help="audit criteria coverage against observed metrics and GenAIRR truth fields")
    audit_parser.add_argument("--report", default=None, help="optional benchmark report JSON")
    audit_parser.add_argument("--cases", default=None, help="optional benchmark case JSONL for truth-field availability")
    audit_parser.add_argument("--out", default=None, help="optional output audit JSON path")
    audit_parser.set_defaults(func=audit)


def resolved_delimiter(value: str | None) -> str | None:
    if value is None:
        return None
    return value.encode("utf-8").decode("unicode_escape")


def load_predictions(path: str, prediction_format: str, delimiter: str | None = None) -> list[dict]:
    if prediction_format == "jsonl":
        return load_dicts_jsonl(path)
    if prediction_format == "airr-tsv":
        return load_airr_predictions(path, delimiter=delimiter or "\t")
    if prediction_format == "airr-csv":
        return load_airr_predictions(path, delimiter=delimiter or ",")
    if prediction_format == "airr":
        return load_airr_predictions(path, delimiter=delimiter)
    raise ValueError(f"unsupported prediction format: {prediction_format}")


def _load_performance_json(path: str | None, *, n_sequences: int) -> dict | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("--performance-json must contain a JSON object")
    return normalize_performance_summary(
        payload,
        n_sequences=n_sequences,
        source=payload.get("source"),
    )


def assay(args) -> None:
    with open(args.path, encoding="utf-8") as handle:
        payload = json.load(handle)
    print(json.dumps(build_assay_report(payload, top_n_contexts=args.top_contexts), indent=2, sort_keys=True))


def evaluate(args) -> None:
    cases = load_jsonl(args.cases)
    predictions = load_predictions(
        args.predictions,
        args.prediction_format,
        delimiter=resolved_delimiter(args.delimiter),
    )
    report = build_benchmark_report(
        cases,
        predictions,
        frame=args.frame,
        contract_level=args.contract_level,
        match_by=None if args.match_by == "order" else args.match_by,
        duplicate_policy=args.duplicate_policy,
        n_bootstrap=args.bootstrap,
        confidence=args.confidence,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_strata=args.bootstrap_strata,
        performance=_load_performance_json(args.performance_json, n_sequences=len(cases)),
    )
    emit_json(report, args.out)


def compare(args) -> None:
    cases = load_jsonl(args.cases)
    prediction_format_a = args.a_prediction_format or args.prediction_format
    prediction_format_b = args.b_prediction_format or args.prediction_format
    delimiter_a = args.a_delimiter if args.a_delimiter is not None else args.delimiter
    delimiter_b = args.b_delimiter if args.b_delimiter is not None else args.delimiter
    predictions_a = load_predictions(
        args.a_predictions,
        prediction_format_a,
        delimiter=resolved_delimiter(delimiter_a),
    )
    predictions_b = load_predictions(
        args.b_predictions,
        prediction_format_b,
        delimiter=resolved_delimiter(delimiter_b),
    )
    report = build_model_comparison_report(
        cases,
        predictions_a,
        predictions_b,
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name,
        frame=args.frame,
        metric_paths=args.metric,
        match_by=None if args.match_by == "order" else args.match_by,
        duplicate_policy=args.duplicate_policy,
        n_bootstrap=args.bootstrap,
        confidence=args.confidence,
        seed=args.bootstrap_seed,
        include_strata=args.bootstrap_strata,
        practical_delta=args.practical_delta,
        case_tie_tolerance=args.case_tie_tolerance,
        comparison_policy=args.policy,
        multiple_comparison_correction=args.multiple_comparison_correction,
        primary_metrics=args.primary_metric,
        guardrail_metrics=args.guardrail_metric,
        minimum_primary_advantage=args.minimum_primary_advantage,
        maximum_guardrail_regression=args.maximum_guardrail_regression,
    )
    emit_json(report, args.out)


def normalize_predictions(args) -> None:
    predictions = load_predictions(
        args.input,
        args.format,
        delimiter=resolved_delimiter(args.delimiter),
    )
    save_dicts_jsonl(predictions, args.out)
    print(
        json.dumps(
            {
                "input": args.input,
                "input_format": args.format,
                "n_predictions": len(predictions),
                "out": args.out,
            },
            indent=2,
            sort_keys=True,
        )
    )


def audit(args) -> None:
    report = None
    if args.report:
        with open(args.report, encoding="utf-8") as handle:
            report = json.load(handle)
    cases = load_jsonl(args.cases) if args.cases else None
    emit_json(audit_criteria_report(report, cases=cases), args.out)
