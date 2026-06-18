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
    coverage_plan_from_spec,
    coverage_summary,
    dataconfig_by_name,
    default_igh_assay_spec,
    default_igh_spec,
    focused_igh_spec,
    generate_benchmark,
    generate_coverage_benchmark,
)
from .io import load_jsonl, save_jsonl
from .core import criteria_catalog, scenario_axes_catalog
from .evaluation import build_assay_report
from ..reference.reference_set import ReferenceSet


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
    generation_report = None
    if args.coverage_planned:
        dataconfig = dataconfig_by_name(spec.dataconfig_name)
        reference_set = ReferenceSet.from_dataconfigs(dataconfig)
        plan = coverage_plan_from_spec(
            spec,
            reference_set,
            min_per_allele=args.min_per_allele,
            min_per_orientation=args.min_per_orientation,
            min_per_context=args.min_per_context,
            min_per_stratum=args.min_per_stratum,
            max_candidates=args.max_candidates,
            name="cli_coverage",
        )
        result = generate_coverage_benchmark(spec, dataconfig, reference_set, plan)
        cases = result.cases
        generation_report = result.report
    else:
        cases = generate_benchmark(spec)
    save_jsonl(cases, args.out)
    payload = {"coverage": coverage_summary(cases)}
    if generation_report is not None:
        payload["generation_coverage"] = generation_report
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {len(cases)} cases to {args.out}")


def _summary(args) -> None:
    cases = load_jsonl(args.path)
    print(json.dumps(coverage_summary(cases), indent=2, sort_keys=True))


def _criteria(args) -> None:
    payload = {"criteria": criteria_catalog(), "scenario_axes": scenario_axes_catalog()}
    print(json.dumps(payload, indent=2, sort_keys=True))


def _assay(args) -> None:
    with open(args.path) as handle:
        payload = json.load(handle)
    print(json.dumps(build_assay_report(payload, top_n_contexts=args.top_contexts), indent=2, sort_keys=True))


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
    build.add_argument("--coverage-planned", action="store_true")
    build.add_argument("--min-per-allele", type=int, default=0)
    build.add_argument("--min-per-orientation", type=int, default=0)
    build.add_argument("--min-per-context", type=int, default=0)
    build.add_argument("--min-per-stratum", type=int, default=0)
    build.add_argument("--max-candidates", type=int, default=None)
    build.set_defaults(func=_build)

    summary = sub.add_parser("summary", help="print coverage summary for a benchmark JSONL")
    summary.add_argument("path")
    summary.set_defaults(func=_summary)

    criteria = sub.add_parser("criteria", help="print the benchmark criteria catalog")
    criteria.set_defaults(func=_criteria)

    assay = sub.add_parser("assay", help="build an assay-style report from score/report JSON")
    assay.add_argument("path")
    assay.add_argument("--top-contexts", type=int, default=25)
    assay.set_defaults(func=_assay)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
