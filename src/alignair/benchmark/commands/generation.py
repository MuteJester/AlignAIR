"""CLI handlers for benchmark generation commands."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

from ..generation import (
    coverage_plan_from_spec,
    coverage_summary,
    dataconfig_by_name,
    default_igh_assay_spec,
    default_igh_spec,
    default_measurement_suite_spec,
    focused_igh_spec,
    generate_benchmark_with_report,
    generate_benchmark_suite,
    generate_coverage_benchmark,
    genotype_subject_spec,
    measurement_aligned_coverage_plan,
    measurement_coverage_summary,
    multi_locus_specs,
    readiness_thresholds,
    restricted_allele_panel_spec,
)
from ..io import export_benchmark_inputs, export_benchmark_suite, save_jsonl
from ...reference.reference_set import ReferenceSet
from .options import READINESS_PROFILE_CHOICES, SUITE_READINESS_PROFILE_CHOICES


def register_generation_commands(subparsers) -> None:
    """Register benchmark generation commands."""

    build_parser = subparsers.add_parser("build", help="generate a benchmark JSONL file")
    build_parser.add_argument("--out", required=True)
    build_parser.add_argument("--config", default="HUMAN_IGH_OGRDB")
    build_parser.add_argument(
        "--recipe",
        choices=("broad", "focused", "assay", "allele-panel", "genotype", "locus"),
        default="broad",
    )
    build_parser.add_argument("--n-per-stratum", type=int, default=200)
    build_parser.add_argument("--n-per-focus", type=int, default=200)
    build_parser.add_argument("--seed", type=int, default=123)
    build_parser.add_argument("--allele-segment", choices=("V", "D", "J", "C"), default="V")
    build_parser.add_argument("--n-panels", type=int, default=3)
    build_parser.add_argument("--alleles-per-panel", type=int, default=2)
    build_parser.add_argument("--n-per-panel", type=int, default=100)
    build_parser.add_argument("--n-subjects", type=int, default=3)
    build_parser.add_argument("--n-per-subject", type=int, default=100)
    build_parser.add_argument("--genotype-seed", type=int, default=1000)
    build_parser.add_argument(
        "--validate-records",
        action="store_true",
        help="run GenAIRR record validation for genotype-subject recipe records",
    )
    build_parser.add_argument(
        "--locus-config",
        action="append",
        default=None,
        help="GenAIRR DataConfig for --recipe locus; repeatable; defaults to IGH, IGK, and IGL",
    )
    build_parser.add_argument("--n-per-locus", type=int, default=100)
    build_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes for benchmark case generation; 1 keeps serial generation",
    )
    build_parser.add_argument("--coverage-planned", action="store_true")
    build_parser.add_argument(
        "--measurement-aligned",
        action="store_true",
        help="when coverage-planned, require measurement-scenario coverage labels",
    )
    build_parser.add_argument("--min-per-measurement-context", type=int, default=1)
    build_parser.add_argument("--min-cases", type=int, default=None)
    build_parser.add_argument("--min-per-allele", type=int, default=0)
    build_parser.add_argument("--min-per-allele-context", type=int, default=0)
    build_parser.add_argument(
        "--allele-context",
        action="append",
        default=None,
        help="coverage label required for each reference allele; repeatable; defaults to profile/spec contexts",
    )
    build_parser.add_argument("--min-per-orientation", type=int, default=0)
    build_parser.add_argument("--min-per-context", type=int, default=0)
    build_parser.add_argument("--min-per-stratum", type=int, default=0)
    build_parser.add_argument("--max-candidates", type=int, default=None)
    build_parser.add_argument("--export-dir", default=None, help="optional directory for FASTA/AIRR/manifest exports")
    build_parser.add_argument("--export-prefix", default="benchmark", help="filename prefix used with --export-dir")
    build_parser.add_argument(
        "--readiness-profile",
        choices=READINESS_PROFILE_CHOICES,
        default="assay",
        help="readiness profile written into exported manifest",
    )
    build_parser.add_argument(
        "--export-frame",
        choices=("presented", "canonical"),
        default="presented",
        help="sequence frame for FASTA and AIRR input exports",
    )
    build_parser.add_argument("--airr-metadata", action="store_true", help="include benchmark metadata in AIRR input TSV")
    build_parser.set_defaults(func=build)

    suite_parser = subparsers.add_parser(
        "build-suite",
        help="generate the composed measurement-aligned benchmark suite",
    )
    suite_parser.add_argument("--out", required=True)
    suite_parser.add_argument("--config", default="HUMAN_IGH_OGRDB")
    suite_parser.add_argument("--seed", type=int, default=123)
    suite_parser.add_argument("--n-per-stratum", type=int, default=200)
    suite_parser.add_argument("--n-per-focus", type=int, default=200)
    suite_parser.add_argument("--n-panels", type=int, default=3)
    suite_parser.add_argument("--alleles-per-panel", type=int, default=2)
    suite_parser.add_argument("--n-per-panel", type=int, default=100)
    suite_parser.add_argument("--n-subjects", type=int, default=3)
    suite_parser.add_argument("--n-per-subject", type=int, default=100)
    suite_parser.add_argument("--genotype-seed", type=int, default=1000)
    suite_parser.add_argument(
        "--validate-records",
        action="store_true",
        help="run GenAIRR record validation for genotype-subject component records",
    )
    suite_parser.add_argument(
        "--locus-config",
        action="append",
        default=None,
        help="GenAIRR DataConfig for the multi-locus component; repeatable",
    )
    suite_parser.add_argument("--n-per-locus", type=int, default=100)
    suite_parser.add_argument("--workers", type=int, default=1)
    suite_parser.add_argument(
        "--suite-readiness-profile",
        choices=SUITE_READINESS_PROFILE_CHOICES,
        default="assay",
        help="measurement-level readiness gate profile attached to the suite report",
    )
    suite_parser.add_argument("--no-base-assay", dest="include_base_assay", action="store_false")
    suite_parser.add_argument("--no-allele-panels", dest="include_allele_panels", action="store_false")
    suite_parser.add_argument("--no-genotype-subjects", dest="include_genotype_subjects", action="store_false")
    suite_parser.add_argument("--no-multi-locus", dest="include_multi_locus", action="store_false")
    suite_parser.set_defaults(
        include_base_assay=True,
        include_allele_panels=True,
        include_genotype_subjects=True,
        include_multi_locus=True,
    )
    suite_parser.add_argument("--export-dir", default=None, help="optional directory for suite export pack")
    suite_parser.add_argument("--export-prefix", default="benchmark_suite", help="suite export filename prefix")
    suite_parser.add_argument(
        "--readiness-profile",
        choices=READINESS_PROFILE_CHOICES,
        default="assay",
        help="readiness profile for the combined exported manifest",
    )
    suite_parser.add_argument(
        "--export-frame",
        choices=("presented", "canonical"),
        default="presented",
        help="sequence frame for FASTA and AIRR input exports",
    )
    suite_parser.add_argument("--airr-metadata", action="store_true", help="include benchmark metadata in AIRR TSVs")
    suite_parser.set_defaults(func=build_suite)


def _build_specs(args) -> tuple:
    if args.recipe == "focused":
        spec = focused_igh_spec(n_per_scenario=args.n_per_focus, seed=args.seed)
    elif args.recipe == "assay":
        spec = default_igh_assay_spec(
            n_per_stratum=args.n_per_stratum,
            n_per_focus=args.n_per_focus,
            seed=args.seed,
        )
    elif args.recipe == "allele-panel":
        spec = restricted_allele_panel_spec(
            dataconfig_name=args.config,
            seed=args.seed,
            segment=args.allele_segment,
            n_panels=args.n_panels,
            alleles_per_panel=args.alleles_per_panel,
            n_per_panel=args.n_per_panel,
        )
    elif args.recipe == "genotype":
        spec = genotype_subject_spec(
            dataconfig_name=args.config,
            seed=args.seed,
            n_subjects=args.n_subjects,
            n_per_subject=args.n_per_subject,
            genotype_seed=args.genotype_seed,
            validate_records=args.validate_records,
        )
    elif args.recipe == "locus":
        configs = tuple(
            args.locus_config or ("HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB", "HUMAN_IGL_OGRDB")
        )
        return multi_locus_specs(
            dataconfig_names=configs,
            seed=args.seed,
            n_per_locus=args.n_per_locus,
        )
    else:
        spec = default_igh_spec(n_per_stratum=args.n_per_stratum, seed=args.seed)

    if args.recipe in {"broad", "focused", "assay"} and args.config != spec.dataconfig_name:
        spec = replace(
            spec,
            name=f"{args.config.lower()}_{args.recipe}",
            dataconfig_name=args.config,
            description=f"{args.recipe.title()} GenAIRR benchmark for {args.config}.",
        )
    return (spec,)


def _reference_set_for_specs(specs) -> ReferenceSet:
    dataconfigs = [dataconfig_by_name(spec.dataconfig_name) for spec in specs]
    return ReferenceSet.from_dataconfigs(*dataconfigs)


def _coverage_plan_for_spec(spec, reference_set, args):
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
    allele_contexts = tuple(args.allele_context or ()) or readiness_threshold.allele_contexts
    if args.measurement_aligned:
        return measurement_aligned_coverage_plan(
            spec,
            reference_set,
            min_cases=min_cases,
            min_per_measurement_context=args.min_per_measurement_context,
            min_per_allele=min_per_allele,
            min_per_orientation=args.min_per_orientation,
            min_per_context=args.min_per_context,
            min_per_stratum=args.min_per_stratum,
            min_per_allele_context=min_per_allele_context,
            allele_contexts=allele_contexts,
            max_candidates=args.max_candidates,
            name="cli_measurement_aligned_coverage",
        )
    return coverage_plan_from_spec(
        spec,
        reference_set,
        min_cases=min_cases,
        min_per_allele=min_per_allele,
        min_per_orientation=args.min_per_orientation,
        min_per_context=args.min_per_context,
        min_per_stratum=args.min_per_stratum,
        min_per_allele_context=min_per_allele_context,
        allele_contexts=allele_contexts,
        max_candidates=args.max_candidates,
        name="cli_coverage",
    )


def _combined_generation_profile(specs, profiles, cases):
    if len(profiles) == 1:
        return profiles[0]
    return {
        "mode": "multi_spec",
        "n_specs": len(profiles),
        "n_cases": len(cases),
        "profiles": [
            {
                "spec": spec.name,
                "dataconfig_name": spec.dataconfig_name,
                "profile": profile,
            }
            for spec, profile in zip(specs, profiles)
        ],
    }


def _combined_coverage_report(specs, reports):
    if not reports:
        return None
    if len(reports) == 1:
        return reports[0]
    return {
        "mode": "multi_spec_coverage_planned",
        "n_specs": len(reports),
        "reports": [
            {
                "spec": spec.name,
                "dataconfig_name": spec.dataconfig_name,
                "coverage": report,
            }
            for spec, report in zip(specs, reports)
        ],
    }


def build(args) -> None:
    """Generate a benchmark JSONL using one recipe or a multi-spec recipe."""

    specs = _build_specs(args)
    cases = []
    profiles = []
    coverage_reports = []
    per_spec_reference_sets = []
    for spec in specs:
        dataconfig = None
        reference_set = None
        if args.coverage_planned or args.export_dir or len(specs) > 1:
            dataconfig = dataconfig_by_name(spec.dataconfig_name)
            reference_set = ReferenceSet.from_dataconfigs(dataconfig)
            per_spec_reference_sets.append(reference_set)
        if args.coverage_planned:
            plan = _coverage_plan_for_spec(spec, reference_set, args)
            result = generate_coverage_benchmark(
                spec,
                dataconfig,
                reference_set,
                plan,
                workers=args.workers,
            )
            cases.extend(result.cases)
            coverage_reports.append(result.report)
            profiles.append(result.report.get("generation_profile"))
        else:
            result = generate_benchmark_with_report(
                spec,
                dataconfig=dataconfig,
                reference_set=reference_set,
                workers=args.workers,
            )
            cases.extend(result.cases)
            profiles.append(result.report)

    generation_profile = _combined_generation_profile(specs, profiles, cases)
    generation_coverage_report = _combined_coverage_report(specs, coverage_reports)
    save_jsonl(cases, args.out)
    payload = {
        "coverage": coverage_summary(cases),
        "measurement_coverage": measurement_coverage_summary(cases),
    }
    if generation_profile is not None:
        payload["generation_profile"] = generation_profile
    if generation_coverage_report is not None:
        payload["generation_coverage"] = generation_coverage_report
    if args.export_dir:
        manifest_generation_report = {
            "profile": generation_profile,
            "specs": [asdict(spec) for spec in specs],
        }
        if generation_coverage_report is not None:
            manifest_generation_report["coverage"] = generation_coverage_report
        manifest_spec = specs[0] if len(specs) == 1 else None
        manifest_dataconfig = (
            specs[0].dataconfig_name
            if len(specs) == 1
            else ",".join(spec.dataconfig_name for spec in specs)
        )
        if len(per_spec_reference_sets) == 1:
            manifest_reference_set = per_spec_reference_sets[0]
        else:
            manifest_reference_set = _reference_set_for_specs(specs)
        payload["exports"] = export_benchmark_inputs(
            cases,
            args.export_dir,
            prefix=args.export_prefix,
            frame=args.export_frame,
            include_airr_metadata=args.airr_metadata,
            spec=manifest_spec,
            dataconfig_name=manifest_dataconfig,
            reference_set=manifest_reference_set,
            generation_report=manifest_generation_report,
            readiness_profile=args.readiness_profile,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {len(cases)} cases to {args.out}")


def build_suite(args) -> None:
    """Generate the composed measurement-aligned benchmark suite."""

    suite = default_measurement_suite_spec(
        dataconfig_name=args.config,
        seed=args.seed,
        n_per_stratum=args.n_per_stratum,
        n_per_focus=args.n_per_focus,
        n_panels=args.n_panels,
        alleles_per_panel=args.alleles_per_panel,
        n_per_panel=args.n_per_panel,
        n_subjects=args.n_subjects,
        n_per_subject=args.n_per_subject,
        genotype_seed=args.genotype_seed,
        validate_records=args.validate_records,
        locus_dataconfig_names=tuple(
            args.locus_config or ("HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB", "HUMAN_IGL_OGRDB")
        ),
        n_per_locus=args.n_per_locus,
        include_base_assay=args.include_base_assay,
        include_allele_panels=args.include_allele_panels,
        include_genotype_subjects=args.include_genotype_subjects,
        include_multi_locus=args.include_multi_locus,
    )
    result = generate_benchmark_suite(
        suite,
        workers=args.workers,
        suite_readiness_profile=args.suite_readiness_profile,
    )
    save_jsonl(result.cases, args.out)
    payload = dict(result.report)
    if args.export_dir:
        payload["exports"] = export_benchmark_suite(
            result,
            args.export_dir,
            prefix=args.export_prefix,
            frame=args.export_frame,
            include_airr_metadata=args.airr_metadata,
            readiness_profile=args.readiness_profile,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {len(result.cases)} suite cases to {args.out}")
