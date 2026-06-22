"""Small CLI for building and inspecting benchmark JSONL files.

Usage:
  PYTHONPATH=src python -m alignair.benchmark.cli build --out experiments/bench.jsonl
  PYTHONPATH=src python -m alignair.benchmark.cli build --recipe assay --coverage-planned --out experiments/bench.jsonl
  PYTHONPATH=src python -m alignair.benchmark.cli summary experiments/bench.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .generation import (
    assess_benchmark_readiness,
    coverage_plan_from_spec,
    coverage_summary,
    dataconfig_by_name,
    default_igh_assay_spec,
    default_igh_spec,
    focused_igh_spec,
    generate_benchmark_with_report,
    generate_coverage_benchmark,
    readiness_thresholds,
)
from .io import (
    export_benchmark_inputs,
    load_airr_predictions,
    load_dicts_jsonl,
    load_jsonl,
    save_dicts_jsonl,
    save_jsonl,
)
from .core import criteria_catalog, scenario_axes_catalog
from .evaluation import (
    audit_criteria_report,
    build_assay_report,
    build_benchmark_report,
    build_model_comparison_report,
    comparison_policy_catalog,
    MULTIPLE_COMPARISON_CORRECTIONS,
    normalize_performance_summary,
    prediction_contract,
)
from ..reference.reference_set import ReferenceSet

PREDICTION_FORMATS = ("jsonl", "airr", "airr-tsv", "airr-csv")
READINESS_PROFILE_CHOICES = ("smoke", "development", "assay", "allele_complete", "allele_stratified")
COMPARISON_POLICY_CHOICES = tuple(row["name"] for row in comparison_policy_catalog())


def _resolved_delimiter(value: str | None) -> str | None:
    if value is None:
        return None
    return value.encode("utf-8").decode("unicode_escape")


def _load_predictions(path: str, prediction_format: str, delimiter: str | None = None) -> list[dict]:
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


def _build(args) -> None:
    if args.recipe == "focused":
        spec = focused_igh_spec(n_per_scenario=args.n_per_focus, seed=args.seed)
    elif args.recipe == "assay":
        spec = default_igh_assay_spec(
            n_per_stratum=args.n_per_stratum,
            n_per_focus=args.n_per_focus,
            seed=args.seed,
        )
    else:
        spec = default_igh_spec(n_per_stratum=args.n_per_stratum, seed=args.seed)
    if args.config != spec.dataconfig_name:
        spec = replace(
            spec,
            name=f"{args.config.lower()}_{args.recipe}",
            dataconfig_name=args.config,
            description=f"{args.recipe.title()} GenAIRR benchmark for {args.config}.",
        )
    dataconfig = None
    reference_set = None
    generation_coverage_report = None
    generation_profile = None
    if args.coverage_planned or args.export_dir:
        dataconfig = dataconfig_by_name(spec.dataconfig_name)
        reference_set = ReferenceSet.from_dataconfigs(dataconfig)
    if args.coverage_planned:
        readiness_threshold = readiness_thresholds(args.readiness_profile)
        min_cases = args.min_cases
        if min_cases is None and readiness_threshold.min_cases > 0:
            min_cases = readiness_threshold.min_cases
        min_per_allele = args.min_per_allele
        if min_per_allele <= 0 and readiness_threshold.min_per_reference_allele > 0:
            min_per_allele = readiness_threshold.min_per_reference_allele
        min_per_allele_context = args.min_per_allele_context
        if min_per_allele_context <= 0 and readiness_threshold.min_per_allele_context > 0:
            min_per_allele_context = readiness_threshold.min_per_allele_context
        plan = coverage_plan_from_spec(
            spec,
            reference_set,
            min_cases=min_cases,
            min_per_allele=min_per_allele,
            min_per_orientation=args.min_per_orientation,
            min_per_context=args.min_per_context,
            min_per_stratum=args.min_per_stratum,
            min_per_allele_context=min_per_allele_context,
            allele_contexts=tuple(args.allele_context or ()) or readiness_threshold.allele_contexts,
            max_candidates=args.max_candidates,
            name="cli_coverage",
        )
        result = generate_coverage_benchmark(
            spec,
            dataconfig,
            reference_set,
            plan,
            workers=args.workers,
        )
        cases = result.cases
        generation_coverage_report = result.report
        generation_profile = result.report.get("generation_profile")
    elif args.export_dir:
        result = generate_benchmark_with_report(
            spec,
            reference_set=reference_set,
            workers=args.workers,
        )
        cases = result.cases
        generation_profile = result.report
    else:
        result = generate_benchmark_with_report(spec, workers=args.workers)
        cases = result.cases
        generation_profile = result.report
    save_jsonl(cases, args.out)
    payload = {"coverage": coverage_summary(cases)}
    if generation_profile is not None:
        payload["generation_profile"] = generation_profile
    if generation_coverage_report is not None:
        payload["generation_coverage"] = generation_coverage_report
    if args.export_dir:
        manifest_generation_report = {"profile": generation_profile}
        if generation_coverage_report is not None:
            manifest_generation_report["coverage"] = generation_coverage_report
        payload["exports"] = export_benchmark_inputs(
            cases,
            args.export_dir,
            prefix=args.export_prefix,
            frame=args.export_frame,
            include_airr_metadata=args.airr_metadata,
            spec=spec,
            dataconfig_name=spec.dataconfig_name,
            reference_set=reference_set,
            generation_report=manifest_generation_report,
            readiness_profile=args.readiness_profile,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {len(cases)} cases to {args.out}")


def _summary(args) -> None:
    cases = load_jsonl(args.path)
    print(json.dumps(coverage_summary(cases), indent=2, sort_keys=True))


def _criteria(args) -> None:
    payload = {"criteria": criteria_catalog(), "scenario_axes": scenario_axes_catalog()}
    print(json.dumps(payload, indent=2, sort_keys=True))


def _contract(args) -> None:
    print(json.dumps({"prediction_contract": prediction_contract()}, indent=2, sort_keys=True))


def _comparison_policies(args) -> None:
    print(json.dumps({"comparison_policies": comparison_policy_catalog()}, indent=2, sort_keys=True))


def _assay(args) -> None:
    with open(args.path) as handle:
        payload = json.load(handle)
    print(json.dumps(build_assay_report(payload, top_n_contexts=args.top_contexts), indent=2, sort_keys=True))


def _evaluate(args) -> None:
    cases = load_jsonl(args.cases)
    predictions = _load_predictions(
        args.predictions,
        args.prediction_format,
        delimiter=_resolved_delimiter(args.delimiter),
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
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


def _compare(args) -> None:
    cases = load_jsonl(args.cases)
    prediction_format_a = args.a_prediction_format or args.prediction_format
    prediction_format_b = args.b_prediction_format or args.prediction_format
    delimiter_a = args.a_delimiter if args.a_delimiter is not None else args.delimiter
    delimiter_b = args.b_delimiter if args.b_delimiter is not None else args.delimiter
    predictions_a = _load_predictions(
        args.a_predictions,
        prediction_format_a,
        delimiter=_resolved_delimiter(delimiter_a),
    )
    predictions_b = _load_predictions(
        args.b_predictions,
        prediction_format_b,
        delimiter=_resolved_delimiter(delimiter_b),
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
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


def _normalize_predictions(args) -> None:
    predictions = _load_predictions(
        args.input,
        args.format,
        delimiter=_resolved_delimiter(args.delimiter),
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


def _export(args) -> None:
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


def _readiness(args) -> None:
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
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


def _audit(args) -> None:
    report = None
    if args.report:
        with open(args.report, encoding="utf-8") as handle:
            report = json.load(handle)
    cases = load_jsonl(args.cases) if args.cases else None
    audit = audit_criteria_report(report, cases=cases)
    text = json.dumps(audit, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="GenAIRR-backed AlignAIR benchmark tools")
    sub = parser.add_subparsers(required=True)
    build = sub.add_parser("build", help="generate a benchmark JSONL file")
    build.add_argument("--out", required=True)
    build.add_argument("--config", default="HUMAN_IGH_OGRDB")
    build.add_argument("--recipe", choices=("broad", "focused", "assay"), default="broad")
    build.add_argument("--n-per-stratum", type=int, default=200)
    build.add_argument("--n-per-focus", type=int, default=200)
    build.add_argument("--seed", type=int, default=123)
    build.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes for benchmark case generation; 1 keeps serial generation",
    )
    build.add_argument("--coverage-planned", action="store_true")
    build.add_argument("--min-cases", type=int, default=None)
    build.add_argument("--min-per-allele", type=int, default=0)
    build.add_argument("--min-per-allele-context", type=int, default=0)
    build.add_argument(
        "--allele-context",
        action="append",
        default=None,
        help="coverage label required for each reference allele; repeatable; defaults to profile/spec contexts",
    )
    build.add_argument("--min-per-orientation", type=int, default=0)
    build.add_argument("--min-per-context", type=int, default=0)
    build.add_argument("--min-per-stratum", type=int, default=0)
    build.add_argument("--max-candidates", type=int, default=None)
    build.add_argument("--export-dir", default=None, help="optional directory for FASTA/AIRR/manifest exports")
    build.add_argument("--export-prefix", default="benchmark", help="filename prefix used with --export-dir")
    build.add_argument(
        "--readiness-profile",
        choices=READINESS_PROFILE_CHOICES,
        default="assay",
        help="readiness profile written into exported manifest",
    )
    build.add_argument(
        "--export-frame",
        choices=("presented", "canonical"),
        default="presented",
        help="sequence frame for FASTA and AIRR input exports",
    )
    build.add_argument("--airr-metadata", action="store_true", help="include benchmark metadata in AIRR input TSV")
    build.set_defaults(func=_build)

    summary = sub.add_parser("summary", help="print coverage summary for a benchmark JSONL")
    summary.add_argument("path")
    summary.set_defaults(func=_summary)

    criteria = sub.add_parser("criteria", help="print the benchmark criteria catalog")
    criteria.set_defaults(func=_criteria)

    contract = sub.add_parser("contract", help="print the normalized prediction contract")
    contract.set_defaults(func=_contract)

    comparison_policies = sub.add_parser("comparison-policies", help="print built-in comparison policy templates")
    comparison_policies.set_defaults(func=_comparison_policies)

    assay = sub.add_parser("assay", help="build an assay-style report from score/report JSON")
    assay.add_argument("path")
    assay.add_argument("--top-contexts", type=int, default=25)
    assay.set_defaults(func=_assay)

    normalize = sub.add_parser(
        "normalize-predictions",
        help="normalize prediction files to benchmark JSONL",
    )
    normalize.add_argument("--input", required=True, help="prediction file to normalize")
    normalize.add_argument(
        "--format",
        choices=PREDICTION_FORMATS,
        default="airr",
        help="input prediction format",
    )
    normalize.add_argument("--delimiter", default=None, help="delimiter override for AIRR tables, e.g. '\\t'")
    normalize.add_argument("--out", required=True, help="normalized prediction JSONL path")
    normalize.set_defaults(func=_normalize_predictions)

    export = sub.add_parser("export", help="export benchmark cases for external alignment tools")
    export.add_argument("--cases", required=True, help="benchmark case JSONL")
    export.add_argument("--out-dir", required=True, help="directory for FASTA/AIRR/manifest files")
    export.add_argument("--prefix", default="benchmark", help="output filename prefix")
    export.add_argument(
        "--frame",
        choices=("presented", "canonical"),
        default="presented",
        help="sequence frame for exported input sequences",
    )
    export.add_argument("--config", default=None, help="optional GenAIRR DataConfig name for reference summary")
    export.add_argument("--airr-metadata", action="store_true", help="include benchmark metadata in AIRR input TSV")
    export.add_argument(
        "--readiness-profile",
        choices=READINESS_PROFILE_CHOICES,
        default="assay",
        help="readiness profile written into exported manifest",
    )
    export.set_defaults(func=_export)

    readiness = sub.add_parser("readiness", help="assess generated benchmark coverage before model evaluation")
    readiness.add_argument("--cases", required=True, help="benchmark case JSONL")
    readiness.add_argument("--config", default=None, help="optional GenAIRR DataConfig name for reference coverage")
    readiness.add_argument(
        "--profile",
        choices=READINESS_PROFILE_CHOICES,
        default="assay",
        help="readiness threshold profile",
    )
    readiness.add_argument("--out", default=None, help="optional output readiness JSON path")
    readiness.set_defaults(func=_readiness)

    audit = sub.add_parser("audit", help="audit criteria coverage against observed metrics and GenAIRR truth fields")
    audit.add_argument("--report", default=None, help="optional benchmark report JSON")
    audit.add_argument("--cases", default=None, help="optional benchmark case JSONL for truth-field availability")
    audit.add_argument("--out", default=None, help="optional output audit JSON path")
    audit.set_defaults(func=_audit)

    evaluate = sub.add_parser("evaluate", help="score prediction files against benchmark cases")
    evaluate.add_argument("--cases", required=True, help="benchmark case JSONL")
    evaluate.add_argument("--predictions", required=True, help="prediction file")
    evaluate.add_argument(
        "--prediction-format",
        choices=PREDICTION_FORMATS,
        default="jsonl",
        help="prediction file format",
    )
    evaluate.add_argument("--delimiter", default=None, help="delimiter override for AIRR tables, e.g. '\\t'")
    evaluate.add_argument("--out", default=None, help="optional output report JSON path")
    evaluate.add_argument(
        "--performance-json",
        default=None,
        help="optional JSON sidecar with runtime/memory stats for externally generated predictions",
    )
    evaluate.add_argument("--frame", choices=("canonical", "presented"), default="canonical")
    evaluate.add_argument("--contract-level", choices=("minimal", "core", "assay"), default=None)
    evaluate.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="number of paired case-bootstrap replicates for confidence intervals; 0 disables",
    )
    evaluate.add_argument("--confidence", type=float, default=0.95, help="bootstrap confidence level")
    evaluate.add_argument("--bootstrap-seed", type=int, default=123, help="bootstrap RNG seed")
    evaluate.add_argument(
        "--no-bootstrap-strata",
        dest="bootstrap_strata",
        action="store_false",
        help="skip per-stratum bootstrap intervals while keeping overall bootstrap intervals",
    )
    evaluate.set_defaults(bootstrap_strata=True)
    evaluate.add_argument(
        "--match-by",
        default="sequence_id",
        help="prediction field used to align rows to cases; use 'order' for list-order scoring",
    )
    evaluate.add_argument(
        "--duplicate-policy",
        choices=("first", "last", "error"),
        default="first",
        help="how to handle duplicate prediction ids when --match-by is not 'order'",
    )
    evaluate.set_defaults(func=_evaluate)

    compare = sub.add_parser("compare", help="compare two prediction files on paired benchmark cases")
    compare.add_argument("--cases", required=True, help="benchmark case JSONL")
    compare.add_argument("--a-predictions", required=True, help="model A prediction file")
    compare.add_argument("--b-predictions", required=True, help="model B prediction file")
    compare.add_argument("--model-a-name", default="model_a", help="label for model A")
    compare.add_argument("--model-b-name", default="model_b", help="label for model B")
    compare.add_argument(
        "--prediction-format",
        choices=PREDICTION_FORMATS,
        default="jsonl",
        help="prediction file format used for both models unless overridden",
    )
    compare.add_argument(
        "--a-prediction-format",
        choices=PREDICTION_FORMATS,
        default=None,
        help="optional model A prediction format override",
    )
    compare.add_argument(
        "--b-prediction-format",
        choices=PREDICTION_FORMATS,
        default=None,
        help="optional model B prediction format override",
    )
    compare.add_argument("--delimiter", default=None, help="delimiter override used for both AIRR tables")
    compare.add_argument("--a-delimiter", default=None, help="optional model A delimiter override")
    compare.add_argument("--b-delimiter", default=None, help="optional model B delimiter override")
    compare.add_argument("--out", default=None, help="optional output comparison JSON path")
    compare.add_argument("--frame", choices=("canonical", "presented"), default="canonical")
    compare.add_argument(
        "--policy",
        choices=COMPARISON_POLICY_CHOICES,
        default=None,
        help="built-in endpoint/guardrail policy template",
    )
    compare.add_argument(
        "--metric",
        action="append",
        default=None,
        help="metric path to compare; repeatable; defaults to the standard comparison metrics",
    )
    compare.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="number of paired case-bootstrap replicates for model-difference intervals; 0 disables",
    )
    compare.add_argument("--confidence", type=float, default=0.95, help="bootstrap confidence level")
    compare.add_argument("--bootstrap-seed", type=int, default=123, help="bootstrap RNG seed")
    compare.add_argument(
        "--multiple-comparison-correction",
        choices=MULTIPLE_COMPARISON_CORRECTIONS,
        default="none",
        help="familywise correction used by endpoint/guardrail decision gates",
    )
    compare.add_argument(
        "--no-bootstrap-strata",
        dest="bootstrap_strata",
        action="store_false",
        help="skip per-stratum comparison sections while keeping the overall paired comparison",
    )
    compare.set_defaults(bootstrap_strata=True)
    compare.add_argument(
        "--practical-delta",
        type=float,
        default=0.0,
        help="minimum direction-adjusted aggregate delta needed for a better/worse verdict",
    )
    compare.add_argument(
        "--case-tie-tolerance",
        type=float,
        default=0.0,
        help="per-case direction-adjusted delta treated as a win instead of a tie",
    )
    compare.add_argument(
        "--primary-metric",
        action="append",
        default=None,
        help="primary endpoint metric path for the policy decision; repeatable",
    )
    compare.add_argument(
        "--guardrail-metric",
        action="append",
        default=None,
        help="no-regression guardrail metric path for the policy decision; repeatable",
    )
    compare.add_argument(
        "--minimum-primary-advantage",
        type=float,
        default=None,
        help="minimum direction-adjusted primary endpoint advantage required to pass the gate",
    )
    compare.add_argument(
        "--maximum-guardrail-regression",
        type=float,
        default=None,
        help="maximum allowed direction-adjusted regression on each guardrail metric",
    )
    compare.add_argument(
        "--match-by",
        default="sequence_id",
        help="prediction field used to align both model outputs to cases; use 'order' for list-order scoring",
    )
    compare.add_argument(
        "--duplicate-policy",
        choices=("first", "last", "error"),
        default="first",
        help="how to handle duplicate prediction ids when --match-by is not 'order'",
    )
    compare.set_defaults(func=_compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
