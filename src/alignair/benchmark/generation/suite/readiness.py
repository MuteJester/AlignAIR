"""Readiness gates for composed benchmark suites."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from ...core.schema import BenchmarkCase
from ..planner import case_coverage_labels
from ..scenarios import MEASUREMENT_SCENARIOS, case_measurement_scenarios, measurement_coverage_summary

_GRADE_RANK = {"pass": 0, "warn": 1, "fail": 2}


@dataclass(frozen=True)
class SuiteMeasurementReadinessThresholds:
    """Thresholds for measurement-by-measurement suite readiness."""

    profile: str
    min_cases_per_measurement: int
    min_explicit_cases_per_targeted_measurement: int
    min_cases_per_required_label: int
    statuses: tuple[str, ...] = ("integrated", "coverage_planned")
    required_measurements: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def suite_measurement_readiness_thresholds(
    profile: str = "assay",
    *,
    required_measurements: tuple[str, ...] = (),
) -> SuiteMeasurementReadinessThresholds:
    """Return named suite measurement-readiness threshold presets."""

    if profile == "smoke":
        return SuiteMeasurementReadinessThresholds(
            profile="smoke",
            min_cases_per_measurement=1,
            min_explicit_cases_per_targeted_measurement=1,
            min_cases_per_required_label=1,
            required_measurements=required_measurements,
            description="Minimal per-measurement sanity gate for smoke suites.",
        )
    if profile == "development":
        return SuiteMeasurementReadinessThresholds(
            profile="development",
            min_cases_per_measurement=10,
            min_explicit_cases_per_targeted_measurement=10,
            min_cases_per_required_label=5,
            required_measurements=required_measurements,
            description="Small per-measurement gate for iterative model development.",
        )
    if profile == "assay":
        return SuiteMeasurementReadinessThresholds(
            profile="assay",
            min_cases_per_measurement=50,
            min_explicit_cases_per_targeted_measurement=50,
            min_cases_per_required_label=25,
            required_measurements=required_measurements,
            description="Professional per-measurement gate before benchmark comparisons.",
        )
    raise ValueError("profile must be one of: smoke, development, assay")


def _worst_grade(grades: Iterable[str]) -> str:
    return max(grades, key=lambda grade: _GRADE_RANK.get(grade, -1), default="pass")


def _targeted_measurement_required(scenario) -> bool:
    return (
        f"measurement:{scenario.name}" in scenario.required_coverage_labels
        or not scenario.stratum_names
    )


def _required_labels_for_scenario(scenario) -> tuple[str, ...]:
    labels = list(scenario.required_coverage_labels)
    labels.extend(f"stratum:{name}" for name in scenario.stratum_names)
    return tuple(label for label in dict.fromkeys(labels) if "{" not in label)


def _scenario_filter(thresholds: SuiteMeasurementReadinessThresholds):
    required = set(thresholds.required_measurements)
    statuses = set(thresholds.statuses)
    for scenario in MEASUREMENT_SCENARIOS:
        if scenario.status not in statuses:
            continue
        if required and scenario.name not in required:
            continue
        yield scenario


def assess_benchmark_suite_readiness(
    cases: Iterable[BenchmarkCase],
    *,
    profile: str = "assay",
    thresholds: SuiteMeasurementReadinessThresholds | None = None,
    required_measurements: tuple[str, ...] = (),
    max_examples: int = 25,
) -> dict[str, Any]:
    """Assess whether a composed suite covers each measured benchmark surface."""

    case_list = list(cases)
    thresholds = thresholds or suite_measurement_readiness_thresholds(
        profile,
        required_measurements=required_measurements,
    )
    label_counts = Counter()
    by_measurement: Counter[str] = Counter()
    explicit_by_measurement: Counter[str] = Counter()
    for case in case_list:
        labels = case_coverage_labels(case)
        label_counts.update(labels)
        by_measurement.update(case_measurement_scenarios(case, statuses=thresholds.statuses))
        explicit = (case.record or {}).get("benchmark_measurement")
        if explicit:
            explicit_by_measurement[explicit] += 1

    scenario_rows = []
    for scenario in _scenario_filter(thresholds):
        required_labels = _required_labels_for_scenario(scenario)
        low_labels = {
            label: int(label_counts.get(label, 0))
            for label in required_labels
            if label_counts.get(label, 0) < thresholds.min_cases_per_required_label
        }
        n_cases = int(by_measurement.get(scenario.name, 0))
        n_explicit = int(explicit_by_measurement.get(scenario.name, 0))
        case_grade = "fail" if n_cases < thresholds.min_cases_per_measurement else "pass"
        label_grade = "fail" if low_labels else "pass"
        explicit_required = _targeted_measurement_required(scenario)
        explicit_grade = (
            "fail"
            if explicit_required
            and n_explicit < thresholds.min_explicit_cases_per_targeted_measurement
            else "pass"
        )
        row_grade = _worst_grade((case_grade, label_grade, explicit_grade))
        scenario_rows.append(
            {
                "name": scenario.name,
                "status": scenario.status,
                "grade": row_grade,
                "criteria": scenario.criteria,
                "metric_keys": scenario.metric_keys,
                "n_cases": n_cases,
                "n_explicit_cases": n_explicit,
                "explicit_targeted_cases_required": explicit_required,
                "thresholds": {
                    "min_cases_per_measurement": thresholds.min_cases_per_measurement,
                    "min_explicit_cases_per_targeted_measurement": (
                        thresholds.min_explicit_cases_per_targeted_measurement
                    ),
                    "min_cases_per_required_label": thresholds.min_cases_per_required_label,
                },
                "checks": {
                    "case_count": case_grade,
                    "explicit_targeted_case_count": explicit_grade,
                    "required_label_counts": label_grade,
                },
                "required_label_counts": {
                    label: int(label_counts.get(label, 0)) for label in required_labels[:max_examples]
                },
                "low_required_labels": dict(list(low_labels.items())[:max_examples]),
                "n_low_required_labels": len(low_labels),
            }
        )

    grade = _worst_grade(row["grade"] for row in scenario_rows)
    grade_counts = Counter(row["grade"] for row in scenario_rows)
    return {
        "grade": grade,
        "profile": thresholds.profile,
        "thresholds": thresholds.to_dict(),
        "n_cases": len(case_list),
        "n_scenarios": len(scenario_rows),
        "grade_counts": dict(sorted(grade_counts.items())),
        "scenarios": scenario_rows,
        "failed_scenarios": [
            row["name"] for row in scenario_rows if row["grade"] == "fail"
        ][:max_examples],
        "measurement_coverage": measurement_coverage_summary(
            case_list,
            statuses=thresholds.statuses,
        ),
    }
