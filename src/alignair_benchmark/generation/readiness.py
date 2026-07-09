"""Readiness checks for generated GenAIRR benchmark case sets."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from alignair.reference.reference_set import ReferenceSet
from ..core.artifacts import BENCHMARK_READINESS_REPORT, artifact_metadata
from ..core.schema import BenchmarkCase, GENES, ORIENTATION_NAMES
from .generate import coverage_summary
from .planner import allele_stratification_contexts, case_coverage_labels, core_context_min_counts
from .scenarios import measurement_required_contexts
from .recipes import default_igh_assay_spec

_GRADE_RANK = {"pass": 0, "warn": 1, "fail": 2}


def _required_contexts() -> tuple[str, ...]:
    labels = set(core_context_min_counts(1)) | set(measurement_required_contexts())
    return tuple(sorted(labels))


@dataclass(frozen=True)
class ReadinessThresholds:
    """Thresholds used to decide whether a benchmark is ready for serious use."""

    profile: str
    min_cases: int = 1
    min_per_stratum: int = 1
    min_per_orientation: int = 0
    required_orientation_ids: tuple[int, ...] = ()
    min_per_required_context: int = 0
    required_contexts: tuple[str, ...] = ()
    min_observed_alleles_per_gene: int = 1
    min_reference_allele_fraction: float = 0.0
    min_per_reference_allele: int = 0
    min_per_allele_context: int = 0
    allele_contexts: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def readiness_thresholds(profile: str = "assay") -> ReadinessThresholds:
    """Return named readiness threshold presets."""

    if profile == "smoke":
        return ReadinessThresholds(
            profile="smoke",
            min_cases=1,
            min_per_stratum=1,
            min_observed_alleles_per_gene=1,
            description="Minimal sanity check for tests and development smoke runs.",
        )
    if profile == "development":
        return ReadinessThresholds(
            profile="development",
            min_cases=100,
            min_per_stratum=10,
            min_per_orientation=5,
            required_orientation_ids=(0, 1),
            min_per_required_context=5,
            required_contexts=_required_contexts(),
            min_observed_alleles_per_gene=10,
            min_reference_allele_fraction=0.05,
            description="Small but useful benchmark for model iteration.",
        )
    if profile == "assay":
        return ReadinessThresholds(
            profile="assay",
            min_cases=1000,
            min_per_stratum=50,
            min_per_orientation=25,
            required_orientation_ids=(0, 1, 2, 3),
            min_per_required_context=25,
            required_contexts=_required_contexts(),
            min_observed_alleles_per_gene=25,
            min_reference_allele_fraction=0.25,
            description="Professional preflight profile before comparing alignment models.",
        )
    if profile == "allele_complete":
        return ReadinessThresholds(
            profile="allele_complete",
            min_cases=10_000,
            min_per_stratum=200,
            min_per_orientation=200,
            required_orientation_ids=(0, 1, 2, 3),
            min_per_required_context=200,
            required_contexts=_required_contexts(),
            min_observed_alleles_per_gene=25,
            min_reference_allele_fraction=1.0,
            min_per_reference_allele=100,
            description=(
                "Allele-complete profile requiring every DataConfig reference allele "
                "to appear repeatedly in GenAIRR truth."
            ),
        )
    if profile == "allele_stratified":
        return ReadinessThresholds(
            profile="allele_stratified",
            min_cases=25_000,
            min_per_stratum=200,
            min_per_orientation=200,
            required_orientation_ids=(0, 1, 2, 3),
            min_per_required_context=200,
            required_contexts=_required_contexts(),
            min_observed_alleles_per_gene=25,
            min_reference_allele_fraction=1.0,
            min_per_reference_allele=100,
            min_per_allele_context=1,
            allele_contexts=allele_stratification_contexts(
                default_igh_assay_spec(n_per_stratum=1, n_per_focus=1)
            ),
            description=(
                "Allele-stratified profile requiring every DataConfig reference allele "
                "to appear across high-value benchmark strata and stress contexts."
            ),
        )
    raise ValueError("profile must be one of: smoke, development, assay, allele_complete, allele_stratified")


def _grade_check(observed: float, *, fail_below: float, warn_below: float | None = None) -> str:
    if observed < fail_below:
        return "fail"
    if warn_below is not None and observed < warn_below:
        return "warn"
    return "pass"


def _worst_grade(grades: Iterable[str]) -> str:
    return max(grades, key=lambda grade: _GRADE_RANK.get(grade, -1), default="pass")


def _check(
    name: str,
    grade: str,
    *,
    observed: Any,
    threshold: Any,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "grade": grade,
        "observed": observed,
        "threshold": threshold,
        "message": message,
        "details": details or {},
    }


def assess_benchmark_readiness(
    cases: Iterable[BenchmarkCase],
    *,
    reference_set: ReferenceSet | None = None,
    profile: str = "assay",
    thresholds: ReadinessThresholds | None = None,
    max_examples: int = 25,
) -> dict[str, Any]:
    """Assess whether generated cases are broad enough for benchmark use.

    The assessment uses only GenAIRR-derived benchmark cases, deterministic
    coverage labels, and optional DataConfig-derived reference allele metadata.
    """

    case_list = list(cases)
    thresholds = thresholds or readiness_thresholds(profile)
    required_allele_contexts = tuple(thresholds.allele_contexts)
    context_counts = Counter()
    allele_counts: dict[str, Counter[str]] = {gene: Counter() for gene in GENES}
    allele_context_counts: dict[str, dict[str, Counter[str]]] = {
        gene: {context: Counter() for context in required_allele_contexts}
        for gene in GENES
    }
    orientation_counts = Counter()
    stratum_counts = Counter()
    for case in case_list:
        labels = set(case_coverage_labels(case))
        context_counts.update(labels)
        present_allele_contexts = labels & set(required_allele_contexts)
        stratum_counts[case.stratum] += 1
        orientation_counts[case.orientation_id] += 1
        for gene, truth in case.genes.items():
            allele_counts[gene].update(truth.calls)
            for context in present_allele_contexts:
                allele_context_counts[gene][context].update(truth.calls)

    checks: list[dict[str, Any]] = []
    n_cases = len(case_list)
    checks.append(
        _check(
            "case_count",
            _grade_check(n_cases, fail_below=thresholds.min_cases),
            observed=n_cases,
            threshold=thresholds.min_cases,
            message="Total generated cases should be large enough to stabilize aggregate metrics.",
        )
    )

    low_strata = {
        name: count
        for name, count in sorted(stratum_counts.items())
        if count < thresholds.min_per_stratum
    }
    checks.append(
        _check(
            "stratum_counts",
            "fail" if low_strata else "pass",
            observed=dict(sorted(stratum_counts.items())),
            threshold=thresholds.min_per_stratum,
            message="Each generated stratum should have enough cases for sliced metrics.",
            details={"low_strata": dict(list(low_strata.items())[:max_examples])},
        )
    )

    orientation_details = {}
    low_orientations = {}
    for oid in thresholds.required_orientation_ids:
        count = orientation_counts.get(oid, 0)
        orientation_details[ORIENTATION_NAMES.get(oid, str(oid))] = count
        if count < thresholds.min_per_orientation:
            low_orientations[ORIENTATION_NAMES.get(oid, str(oid))] = count
    if thresholds.required_orientation_ids:
        checks.append(
            _check(
                "orientation_counts",
                "fail" if low_orientations else "pass",
                observed=orientation_details,
                threshold=thresholds.min_per_orientation,
                message="Required orientation transforms should be represented.",
                details={"low_orientations": low_orientations},
            )
        )

    missing_contexts = {}
    if thresholds.required_contexts:
        for label in thresholds.required_contexts:
            count = context_counts.get(label, 0)
            if count < thresholds.min_per_required_context:
                missing_contexts[label] = count
        checks.append(
            _check(
                "required_context_counts",
                "fail" if missing_contexts else "pass",
                observed={
                    label: context_counts.get(label, 0)
                    for label in thresholds.required_contexts[:max_examples]
                },
                threshold=thresholds.min_per_required_context,
                message="Core biological/stress contexts should be covered before model comparison.",
                details={
                    "n_low_contexts": len(missing_contexts),
                    "low_contexts": dict(list(missing_contexts.items())[:max_examples]),
                },
            )
        )

    allele_details = {}
    low_allele_genes = {}
    for gene in GENES:
        observed = len(allele_counts[gene])
        effective_threshold = thresholds.min_observed_alleles_per_gene
        n_reference = None
        if reference_set is not None and gene.upper() in reference_set.genes:
            n_reference = len(reference_set.gene(gene.upper()).names)
            effective_threshold = min(effective_threshold, n_reference)
        allele_details[gene] = {
            "n_observed": observed,
            "threshold": effective_threshold,
            "n_reference": n_reference,
        }
        if observed < effective_threshold:
            low_allele_genes[gene] = observed
    checks.append(
        _check(
            "observed_allele_counts",
            "fail" if low_allele_genes else "pass",
            observed=allele_details,
            threshold=thresholds.min_observed_alleles_per_gene,
            message="Each gene class should cover enough distinct GenAIRR truth alleles.",
            details={
                "low_genes": low_allele_genes,
                "note": "Per-gene thresholds are capped by the reference allele count when a reference set is supplied.",
            },
        )
    )

    if reference_set is not None and thresholds.min_reference_allele_fraction > 0:
        low_reference_fraction = {}
        reference_details = {}
        for gene in GENES:
            ref = reference_set.genes.get(gene.upper())
            if ref is None:
                continue
            total = len(ref.names)
            observed = len(set(ref.names) & set(allele_counts[gene]))
            fraction = observed / total if total else 1.0
            reference_details[gene] = {
                "n_reference": total,
                "n_observed": observed,
                "fraction_observed": fraction,
            }
            if fraction < thresholds.min_reference_allele_fraction:
                low_reference_fraction[gene] = fraction
        checks.append(
            _check(
                "reference_allele_fraction",
                "fail" if low_reference_fraction else "pass",
                observed=reference_details,
                threshold=thresholds.min_reference_allele_fraction,
                message="Observed truth alleles should cover a meaningful fraction of the DataConfig reference set.",
                details={"low_genes": low_reference_fraction},
            )
        )

    if reference_set is not None and thresholds.min_per_reference_allele > 0:
        reference_min_details = {}
        low_reference_alleles: dict[str, dict[str, int]] = {}
        for gene in GENES:
            ref = reference_set.genes.get(gene.upper())
            if ref is None:
                continue
            counts = {name: int(allele_counts[gene].get(name, 0)) for name in ref.names}
            low = {
                name: count
                for name, count in sorted(counts.items())
                if count < thresholds.min_per_reference_allele
            }
            values = list(counts.values())
            reference_min_details[gene] = {
                "n_reference": len(counts),
                "n_passing": len(counts) - len(low),
                "n_low": len(low),
                "min_count": min(values) if values else 0,
                "max_count": max(values) if values else 0,
                "mean_count": (sum(values) / len(values)) if values else 0.0,
            }
            if low:
                low_reference_alleles[gene] = dict(list(low.items())[:max_examples])
        checks.append(
            _check(
                "reference_allele_min_counts",
                "fail" if low_reference_alleles else "pass",
                observed=reference_min_details,
                threshold=thresholds.min_per_reference_allele,
                message=(
                    "Every DataConfig reference allele should appear enough times "
                    "in GenAIRR truth for allele-level model comparison."
                ),
                details={
                    "n_low_alleles": {
                        gene: details["n_low"] for gene, details in reference_min_details.items()
                    },
                    "low_allele_examples": low_reference_alleles,
                },
            )
        )

    if reference_set is not None and thresholds.min_per_allele_context > 0:
        contexts = required_allele_contexts or tuple(
            sorted(label for label in context_counts if label.startswith("stratum:"))
        )
        if not required_allele_contexts:
            allele_context_counts = {
                gene: {context: Counter() for context in contexts}
                for gene in GENES
            }
            for case in case_list:
                labels = set(case_coverage_labels(case))
                present_allele_contexts = labels & set(contexts)
                for gene, truth in case.genes.items():
                    for context in present_allele_contexts:
                        allele_context_counts[gene][context].update(truth.calls)

        context_min_details = {}
        low_cell_examples: dict[str, list[dict[str, Any]]] = {}
        worst_contexts: dict[str, list[dict[str, Any]]] = {}
        for gene in GENES:
            ref = reference_set.genes.get(gene.upper())
            if ref is None:
                continue
            n_cells = len(ref.names) * len(contexts)
            n_low = 0
            n_passing = 0
            min_count = None
            max_count = 0
            context_rows = []
            gene_examples = []
            for context in contexts:
                counts = {
                    name: int(allele_context_counts[gene].get(context, Counter()).get(name, 0))
                    for name in ref.names
                }
                values = list(counts.values())
                context_low = {
                    name: count
                    for name, count in sorted(counts.items())
                    if count < thresholds.min_per_allele_context
                }
                n_low += len(context_low)
                n_passing += len(counts) - len(context_low)
                if values:
                    context_min = min(values)
                    min_count = context_min if min_count is None else min(min_count, context_min)
                    max_count = max(max_count, max(values))
                context_rows.append(
                    {
                        "context": context,
                        "n_low": len(context_low),
                        "min_count": min(values) if values else 0,
                        "n_reference": len(counts),
                    }
                )
                if context_low and len(gene_examples) < max_examples:
                    for allele, count in context_low.items():
                        gene_examples.append({"allele": allele, "context": context, "count": count})
                        if len(gene_examples) >= max_examples:
                            break
            context_min_details[gene] = {
                "n_reference": len(ref.names),
                "n_contexts": len(contexts),
                "n_cells": n_cells,
                "n_passing": n_passing,
                "n_low": n_low,
                "min_count": min_count if min_count is not None else 0,
                "max_count": max_count,
            }
            if gene_examples:
                low_cell_examples[gene] = gene_examples
            worst_contexts[gene] = sorted(
                context_rows,
                key=lambda row: (row["n_low"], -row["min_count"], row["context"]),
                reverse=True,
            )[:max_examples]
        checks.append(
            _check(
                "reference_allele_context_min_counts",
                "fail" if low_cell_examples else "pass",
                observed=context_min_details,
                threshold=thresholds.min_per_allele_context,
                message=(
                    "Every DataConfig reference allele should appear across required "
                    "benchmark contexts, not only somewhere in the aggregate cohort."
                ),
                details={
                    "n_contexts": len(contexts),
                    "contexts": list(contexts[:max_examples]),
                    "n_low_cells": {
                        gene: details["n_low"] for gene, details in context_min_details.items()
                    },
                    "low_cell_examples": low_cell_examples,
                    "worst_contexts": worst_contexts,
                    "note": (
                        "Cells are allele/context intersections; this is axis-marginal "
                        "coverage, not the full Cartesian product of all scenario axes."
                    ),
                },
            )
        )

    grade = _worst_grade(check["grade"] for check in checks)
    grade_counts = Counter(check["grade"] for check in checks)
    return {
        "artifact": artifact_metadata(BENCHMARK_READINESS_REPORT),
        "grade": grade,
        "profile": thresholds.profile,
        "thresholds": thresholds.to_dict(),
        "n_cases": n_cases,
        "grade_counts": dict(sorted(grade_counts.items())),
        "checks": checks,
        "coverage": coverage_summary(case_list),
        "observed_required_context_counts": {
            label: context_counts.get(label, 0)
            for label in thresholds.required_contexts[:max_examples]
        },
        "truth_source": "GenAIRR benchmark cases",
    }
