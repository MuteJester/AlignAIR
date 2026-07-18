"""Benchmark suite generation and summary helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...core.schema import BenchmarkCase
from ..generate import coverage_summary, generate_benchmark_with_report
from ..scenarios import measurement_coverage_summary
from .readiness import assess_benchmark_suite_readiness
from .spec import BenchmarkSuiteSpec, SuiteComponentSpec


@dataclass
class SuiteComponentResult:
    """Generated cases and reports for one suite component."""

    component: SuiteComponentSpec
    cases: list[BenchmarkCase]
    spec_reports: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.to_dict(),
            "n_cases": len(self.cases),
            "coverage": coverage_summary(self.cases),
            "measurement_coverage": measurement_coverage_summary(self.cases),
            "spec_reports": self.spec_reports,
        }


@dataclass
class BenchmarkSuiteResult:
    """Materialized benchmark suite plus component-level generation reports."""

    suite: BenchmarkSuiteSpec
    cases: list[BenchmarkCase]
    components: tuple[SuiteComponentResult, ...]
    report: dict[str, Any]


def _stamp_suite_case(
    case: BenchmarkCase,
    *,
    suite: BenchmarkSuiteSpec,
    component: SuiteComponentSpec,
) -> None:
    case.record = dict(case.record or {})
    case.record["benchmark_suite"] = suite.name
    case.record["benchmark_component"] = component.name
    case.record["benchmark_component_role"] = component.role
    if component.measurement_focus:
        case.record["benchmark_component_measurements"] = ",".join(component.measurement_focus)

    case.tags = dict(case.tags or {})
    case.tags["benchmark_suite"] = suite.name
    case.tags["benchmark_component"] = component.name
    case.tags["benchmark_component_role"] = component.role


def _component_generation_report(component: SuiteComponentSpec, spec_reports) -> dict[str, Any]:
    return {
        "component": component.to_dict(),
        "n_specs": len(component.specs),
        "n_cases": sum(
            report.get("generation_profile", {}).get("n_cases", 0) for report in spec_reports
        ),
        "spec_reports": tuple(spec_reports),
    }


def benchmark_suite_summary(
    suite: BenchmarkSuiteSpec,
    cases: list[BenchmarkCase],
    components: tuple[SuiteComponentResult, ...],
) -> dict[str, Any]:
    """Return a serializable high-level summary for a generated suite."""

    return {
        "suite": suite.to_dict(),
        "n_cases": len(cases),
        "components": [component.to_dict() for component in components],
        "component_case_counts": {
            component.component.name: len(component.cases) for component in components
        },
        "coverage": coverage_summary(cases),
        "measurement_coverage": measurement_coverage_summary(cases),
    }


def _required_measurements_for_suite(suite: BenchmarkSuiteSpec) -> tuple[str, ...]:
    if any(component.role == "base_assay" for component in suite.components):
        return ()
    names: list[str] = []
    for component in suite.components:
        names.extend(component.measurement_focus)
    return tuple(dict.fromkeys(names))


def generate_benchmark_suite(
    suite: BenchmarkSuiteSpec,
    *,
    workers: int = 1,
    suite_readiness_profile: str = "assay",
    required_measurements: tuple[str, ...] | None = None,
) -> BenchmarkSuiteResult:
    """Generate every component in a composed benchmark suite."""

    all_cases: list[BenchmarkCase] = []
    component_results: list[SuiteComponentResult] = []
    for component in suite.components:
        component_cases: list[BenchmarkCase] = []
        spec_reports: list[dict[str, Any]] = []
        for spec in component.specs:
            result = generate_benchmark_with_report(spec, workers=workers)
            for case in result.cases:
                _stamp_suite_case(case, suite=suite, component=component)
            component_cases.extend(result.cases)
            all_cases.extend(result.cases)
            spec_reports.append(
                {
                    "spec": asdict(spec),
                    "generation_profile": result.report,
                }
            )
        component_results.append(
            SuiteComponentResult(
                component=component,
                cases=component_cases,
                spec_reports=tuple(spec_reports),
            )
        )

    components_tuple = tuple(component_results)
    report = benchmark_suite_summary(suite, all_cases, components_tuple)
    report["suite_readiness"] = assess_benchmark_suite_readiness(
        all_cases,
        profile=suite_readiness_profile,
        required_measurements=(
            _required_measurements_for_suite(suite)
            if required_measurements is None
            else required_measurements
        ),
    )
    report["generation_report"] = {
        "mode": "benchmark_suite",
        "n_components": len(components_tuple),
        "components": [
            _component_generation_report(component.component, component.spec_reports)
            for component in components_tuple
        ],
    }
    return BenchmarkSuiteResult(
        suite=suite,
        cases=all_cases,
        components=components_tuple,
        report=report,
    )
