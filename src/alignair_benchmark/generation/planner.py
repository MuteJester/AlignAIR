"""Coverage-driven benchmark generation.

The plain generator follows fixed stratum counts. This module adds an online
planner that keeps accepting sampled cases until requested coverage quotas are
met, for example a minimum count for each orientation, ambiguity class, or
reference allele.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Iterator

from alignair.reference.reference_set import ReferenceSet
from ..core.schema import BenchmarkCase, BenchmarkSpec, GENES, ORIENTATION_NAMES
from .generate import (
    _case_from_record,
    _resolved_workers,
    candidate_round_worker,
    dataconfig_by_name,
    generation_run_report,
)
from .genairr import apply_benchmark_crop, build_stratum_experiment, stream_stratum_records

ALLELE_CONTEXT_PREFIX = "allele_context:"


def allele_context_label(gene: str, allele: str, context_label: str) -> str:
    """Return the synthetic coverage label for an allele/context intersection."""

    return f"{ALLELE_CONTEXT_PREFIX}{gene}:{allele}|{context_label}"


def _bin(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def case_coverage_labels(case: BenchmarkCase) -> tuple[str, ...]:
    """Return stable coverage labels contributed by one benchmark case."""

    record = case.record or {}
    tags = case.tags or {}
    labels: list[str] = [f"stratum:{case.stratum}"]
    labels.extend(f"tag:{tag}" for tag in tags.get("stratum_tags", ()))
    if record.get("benchmark_measurement"):
        labels.append(f"measurement:{record['benchmark_measurement']}")
    if record.get("locus"):
        labels.append(f"locus:{record['locus']}")
    labels.append("chain:has_d" if case.genes.get("d") and case.genes["d"].calls else "chain:no_d")
    labels.append(f"orientation:{ORIENTATION_NAMES.get(case.orientation_id, case.orientation_id)}")
    labels.append(
        "length:"
        + _bin(len(case.sequence), (60, 90, 130, 250), ("<=60", "61-90", "91-130", "131-250", ">250"))
    )
    labels.append(
        "mutation:"
        + _bin(
            float(tags.get("mutation_rate", 0.0)),
            (0.01, 0.05, 0.12, 0.18),
            ("<=1%", "1-5%", "5-12%", "12-18%", ">18%"),
        )
    )
    labels.append("indel:" + _bin(float(tags.get("n_indels", 0.0)), (0, 2, 5), ("0", "1-2", "3-5", ">5")))
    noise = float(tags.get("n_quality_errors", 0) or 0) + float(tags.get("n_pcr_errors", 0) or 0)
    labels.append("noise:" + _bin(noise, (0, 2, 8), ("0", "1-2", "3-8", ">8")))
    labels.append("productivity:yes" if tags.get("productive") else "productivity:no")

    if case.genes.get("d") and case.genes["d"].calls:
        labels.append("d_orientation:inverted" if tags.get("d_inverted") else "d_orientation:forward")
    else:
        labels.append("d_orientation:not_applicable")

    layout = record.get("read_layout") or "single"
    labels.append(f"read_layout:{layout}")
    labels.append("contaminant:yes" if record.get("is_contaminant") else "contaminant:no")
    labels.append("revision:yes" if record.get("receptor_revision_applied") else "revision:no")
    labels.append("constant_region:present" if record.get("c_call") else "constant_region:absent")

    junction_len = int(record.get("junction_length") or 0)
    labels.append(
        "junction_length:"
        + _bin(
            junction_len,
            (30, 75, 120),
            ("short_junction", "typical_junction", "long_junction", "very_long_junction"),
        )
    )
    labels.append("junction_frame:in_frame" if record.get("vj_in_frame") else "junction_frame:out_of_frame")
    if record.get("stop_codon"):
        labels.append("junction_frame:stop_codon")

    v = case.genes.get("v")
    d = case.genes.get("d")
    j = case.genes.get("j")
    segment_labels = []
    if v and v.sequence_start is not None and v.sequence_end is not None and (v.sequence_end - v.sequence_start) < 80:
        segment_labels.append("short_v_tail")
    if d and d.sequence_start is not None and d.sequence_end is not None and (d.sequence_end - d.sequence_start) < 8:
        segment_labels.append("short_d")
    if j and j.sequence_start is not None and j.sequence_end is not None and (j.sequence_end - j.sequence_start) < 20:
        segment_labels.append("short_j_head")
    if not segment_labels:
        segment_labels.append("all_segments_visible")
    labels.extend(f"segment_presence:{name}" for name in segment_labels)

    any_multi = False
    for gene, truth in case.genes.items():
        if len(truth.calls) > 1:
            any_multi = True
            labels.append(f"ambiguity:{gene}_multi")
        else:
            labels.append(f"ambiguity:{gene}_single")
        for call in truth.calls:
            labels.append(f"allele:{gene}:{call}")
            labels.append(f"gene:{gene}:{call.split('*')[0]}")
    labels.append("ambiguity:any_multi" if any_multi else "ambiguity:all_single")
    return tuple(labels)


def allele_stratification_contexts(spec: BenchmarkSpec) -> tuple[str, ...]:
    """Return high-value context labels for per-allele matrix coverage.

    This intentionally uses axis-marginal contexts instead of the full Cartesian
    product of all contexts. It asks whether each allele appears under each
    important benchmark scenario axis, without requiring every allele under
    every possible combination of axes.
    """

    labels: list[str] = []
    labels.extend(f"stratum:{stratum.name}" for stratum in spec.strata)
    orientation_ids = sorted({oid for stratum in spec.strata for oid in (stratum.orientation_ids or (0,))})
    labels.extend(f"orientation:{ORIENTATION_NAMES.get(oid, oid)}" for oid in orientation_ids)
    labels.extend(
        [
            "tag:full",
            "tag:fragment",
            "tag:orientation",
            "tag:hard",
            "tag:extreme",
            "tag:shm",
            "tag:indel",
            "tag:noise",
            "tag:trim",
            "tag:adaptive",
            "tag:fr1",
            "tag:fr2",
            "tag:fr3",
            "tag:j_anchored",
            "tag:short",
            "tag:contaminant",
            "tag:d_inversion",
            "tag:read_layout",
            "tag:receptor_revision",
            "ambiguity:all_single",
            "ambiguity:any_multi",
            "segment_presence:all_segments_visible",
            "segment_presence:short_v_tail",
            "segment_presence:short_d",
            "segment_presence:short_j_head",
            "length:<=60",
            "length:61-90",
            "length:91-130",
            "length:131-250",
            "length:>250",
            "mutation:<=1%",
            "mutation:1-5%",
            "mutation:5-12%",
            "mutation:12-18%",
            "mutation:>18%",
            "indel:0",
            "indel:1-2",
            "indel:3-5",
            "indel:>5",
            "noise:0",
            "noise:1-2",
            "noise:3-8",
            "noise:>8",
            "junction_frame:in_frame",
            "junction_frame:out_of_frame",
            "junction_frame:stop_codon",
        ]
    )
    return tuple(dict.fromkeys(labels))


def _allele_context_targets(min_counts: dict[str, int]) -> set[str]:
    contexts = set()
    for label in min_counts:
        if label.startswith(ALLELE_CONTEXT_PREFIX) and "|" in label:
            contexts.add(label.split("|", 1)[1])
    return contexts


def _labels_for_plan(case: BenchmarkCase, target_contexts: set[str] | None = None) -> tuple[str, ...]:
    labels = list(case_coverage_labels(case))
    if not target_contexts:
        return tuple(labels)

    present_contexts = set(labels) & target_contexts
    if not present_contexts:
        return tuple(labels)
    for gene, truth in case.genes.items():
        for call in truth.calls:
            labels.extend(
                allele_context_label(gene, call, context_label)
                for context_label in present_contexts
            )
    return tuple(labels)


@dataclass(frozen=True)
class CoveragePlan:
    """Coverage targets for online benchmark generation.

    ``min_counts`` maps coverage labels, such as ``orientation:reverse`` or
    ``allele:v:IGHV1-2*01``, to required accepted-case counts.
    """

    name: str = "coverage"
    min_cases: int | None = None
    min_counts: dict[str, int] = field(default_factory=dict)
    max_candidates: int | None = None
    accept_all_until_min_cases: bool = True
    stop_when_satisfied: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["min_counts"] = dict(sorted(self.min_counts.items()))
        return data


@dataclass
class CoverageGenerationResult:
    """Materialized planned benchmark plus coverage-generation report."""

    cases: list[BenchmarkCase]
    report: dict[str, Any]


class CoverageTracker:
    """Mutable coverage ledger used by coverage-driven streams."""

    def __init__(self, plan: CoveragePlan) -> None:
        self.plan = plan
        self.generated_cases = 0
        self.accepted_cases = 0
        self.counts: Counter[str] = Counter()
        self.accepted_by_stratum: Counter[str] = Counter()
        self._target_contexts = _allele_context_targets(plan.min_counts)

    @property
    def min_cases(self) -> int:
        return int(self.plan.min_cases or 0)

    @property
    def unmet(self) -> dict[str, int]:
        missing = {}
        for label, target in self.plan.min_counts.items():
            deficit = int(target) - self.counts.get(label, 0)
            if deficit > 0:
                missing[label] = deficit
        if self.accepted_cases < self.min_cases:
            missing["__min_cases__"] = self.min_cases - self.accepted_cases
        return dict(sorted(missing.items()))

    @property
    def satisfied(self) -> bool:
        return not self.unmet

    def record_candidate(self) -> None:
        self.generated_cases += 1

    def missing_labels_for(self, case: BenchmarkCase) -> tuple[str, ...]:
        labels = _labels_for_plan(case, self._target_contexts)
        return tuple(
            label
            for label in labels
            if label in self.plan.min_counts and self.counts.get(label, 0) < self.plan.min_counts[label]
        )

    def should_accept(self, case: BenchmarkCase) -> bool:
        if self.plan.accept_all_until_min_cases and self.accepted_cases < self.min_cases:
            return True
        return bool(self.missing_labels_for(case))

    def accept(self, case: BenchmarkCase) -> None:
        self.accepted_cases += 1
        self.accepted_by_stratum[case.stratum] += 1
        self.counts.update(_labels_for_plan(case, self._target_contexts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "generated_cases": self.generated_cases,
            "accepted_cases": self.accepted_cases,
            "accepted_by_stratum": dict(sorted(self.accepted_by_stratum.items())),
            "satisfied": self.satisfied,
            "unmet": self.unmet,
            "observed_target_counts": {
                label: self.counts.get(label, 0) for label in sorted(self.plan.min_counts)
            },
            "observed_counts": dict(sorted(self.counts.items())),
        }


def core_context_min_counts(min_count: int) -> dict[str, int]:
    """Return useful non-allele context quotas for broad AIRR benchmark builds."""

    if min_count <= 0:
        return {}
    labels = [
        "productivity:yes",
        "productivity:no",
        "d_orientation:forward",
        "d_orientation:inverted",
        "ambiguity:all_single",
        "ambiguity:any_multi",
        "length:<=60",
        "length:61-90",
        "length:91-130",
        "length:131-250",
        "length:>250",
        "mutation:<=1%",
        "mutation:1-5%",
        "mutation:5-12%",
        "mutation:12-18%",
        "mutation:>18%",
        "indel:0",
        "indel:1-2",
        "indel:3-5",
        "indel:>5",
        "noise:0",
        "noise:1-2",
        "noise:3-8",
        "noise:>8",
        "junction_length:short_junction",
        "junction_length:typical_junction",
        "junction_length:long_junction",
        "junction_frame:in_frame",
        "junction_frame:out_of_frame",
        "junction_frame:stop_codon",
        "segment_presence:all_segments_visible",
        "segment_presence:short_v_tail",
        "segment_presence:short_d",
        "segment_presence:short_j_head",
        "read_layout:single",
        "read_layout:paired_end",
        "contaminant:no",
        "contaminant:yes",
        "revision:no",
        "revision:yes",
        "constant_region:absent",
    ]
    return {label: min_count for label in labels}


def coverage_plan_from_spec(
    spec: BenchmarkSpec,
    reference_set: ReferenceSet | None = None,
    *,
    min_cases: int | None = None,
    min_per_allele: int = 0,
    min_per_orientation: int = 0,
    min_per_context: int = 0,
    min_per_stratum: int = 0,
    min_per_allele_context: int = 0,
    allele_contexts: tuple[str, ...] | None = None,
    required_labels: dict[str, int] | None = None,
    max_candidates: int | None = None,
    name: str = "coverage",
) -> CoveragePlan:
    """Build a quota plan from a benchmark spec and optional reference set."""

    min_counts: dict[str, int] = {}
    if min_per_stratum > 0:
        for stratum in spec.strata:
            min_counts[f"stratum:{stratum.name}"] = min_per_stratum
    if min_per_orientation > 0:
        orientation_ids = sorted({oid for stratum in spec.strata for oid in (stratum.orientation_ids or (0,))})
        for oid in orientation_ids:
            min_counts[f"orientation:{ORIENTATION_NAMES.get(oid, oid)}"] = min_per_orientation
    if min_per_context > 0:
        min_counts.update(core_context_min_counts(min_per_context))
    if reference_set is not None and min_per_allele > 0:
        for gene in GENES:
            ref = reference_set.genes.get(gene.upper())
            if ref is None:
                continue
            for name_ in ref.names:
                min_counts[f"allele:{gene}:{name_}"] = min_per_allele
    if reference_set is not None and min_per_allele_context > 0:
        contexts = tuple(allele_contexts or allele_stratification_contexts(spec))
        for gene in GENES:
            ref = reference_set.genes.get(gene.upper())
            if ref is None:
                continue
            for name_ in ref.names:
                for context_label in contexts:
                    min_counts[allele_context_label(gene, name_, context_label)] = min_per_allele_context
    if required_labels:
        for label, count in required_labels.items():
            min_counts[label] = max(int(count), min_counts.get(label, 0))

    target_cases = spec.n_cases if min_cases is None else int(min_cases)
    if max_candidates is None:
        quota_mass = sum(min_counts.values())
        max_candidates = max(target_cases, quota_mass, spec.n_cases, 1) * 20 + 100
    return CoveragePlan(
        name=name,
        min_cases=target_cases,
        min_counts=min_counts,
        max_candidates=max_candidates,
        description="Coverage-driven GenAIRR benchmark plan.",
    )


def _default_max_candidates(spec: BenchmarkSpec, plan: CoveragePlan) -> int:
    target_cases = int(plan.min_cases or spec.n_cases)
    quota_mass = sum(plan.min_counts.values())
    return max(target_cases, quota_mass, spec.n_cases, 1) * 20 + 100


def _candidate_cases(
    spec: BenchmarkSpec,
    dataconfig,
    reference_set: ReferenceSet,
    *,
    max_candidates: int,
) -> Iterator[BenchmarkCase]:
    generated = 0
    round_idx = 0
    active_strata = tuple(
        (s_idx, stratum) for s_idx, stratum in enumerate(spec.strata) if stratum.n > 0
    )
    if not active_strata:
        return
    while generated < max_candidates:
        for s_idx, stratum in active_strata:
            resolved = build_stratum_experiment(dataconfig, stratum)
            seed = spec.seed + stratum.seed_offset + 1009 * s_idx + 1_000_003 * round_idx
            for i, record in enumerate(stream_stratum_records(resolved, n=stratum.n, seed=seed)):
                if generated >= max_candidates:
                    return
                record = apply_benchmark_crop(record, stratum)
                orientations = stratum.orientation_ids or (0,)
                stratum_index = round_idx * stratum.n + i
                orientation_id = orientations[stratum_index % len(orientations)]
                if round_idx == 0:
                    case_id = f"{spec.name}:{stratum.name}:{i:06d}"
                else:
                    case_id = f"{spec.name}:{stratum.name}:extra{round_idx:03d}:{i:06d}"
                generated += 1
                yield _case_from_record(record, reference_set, stratum, case_id, orientation_id)
        round_idx += 1


def _candidate_cases_parallel(
    spec: BenchmarkSpec,
    reference_set: ReferenceSet,
    *,
    max_candidates: int,
    workers: int,
) -> Iterator[BenchmarkCase]:
    """Yield candidate cases from parallel stratum/round chunks in deterministic order."""

    generated = 0
    round_idx = 0
    active_strata = tuple(
        (s_idx, stratum) for s_idx, stratum in enumerate(spec.strata) if stratum.n > 0
    )
    if not active_strata:
        return
    max_workers = min(workers, len(active_strata))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        while generated < max_candidates:
            tasks = [
                (spec, s_idx, stratum, round_idx, reference_set)
                for s_idx, stratum in active_strata
            ]
            results = list(pool.map(candidate_round_worker, tasks))
            for _, cases in sorted(results, key=lambda item: item[0]):
                for case in cases:
                    if generated >= max_candidates:
                        return
                    generated += 1
                    yield case
            round_idx += 1


def stream_coverage_benchmark(
    spec: BenchmarkSpec,
    dataconfig=None,
    reference_set: ReferenceSet | None = None,
    plan: CoveragePlan | None = None,
    tracker: CoverageTracker | None = None,
    workers: int = 1,
) -> Iterator[BenchmarkCase]:
    """Yield accepted cases until the coverage plan is satisfied or exhausted."""

    dataconfig = dataconfig or dataconfig_by_name(spec.dataconfig_name)
    reference_set = reference_set or ReferenceSet.from_dataconfigs(dataconfig)
    plan = plan or coverage_plan_from_spec(spec, reference_set)
    tracker = tracker or CoverageTracker(plan)
    max_candidates = int(plan.max_candidates or _default_max_candidates(spec, plan))
    resolved_workers = _resolved_workers(workers)

    if resolved_workers > 1:
        candidate_iter = _candidate_cases_parallel(
            spec,
            reference_set,
            max_candidates=max_candidates,
            workers=resolved_workers,
        )
    else:
        candidate_iter = _candidate_cases(
            spec,
            dataconfig,
            reference_set,
            max_candidates=max_candidates,
        )

    for case in candidate_iter:
        tracker.record_candidate()
        if tracker.should_accept(case):
            tracker.accept(case)
            yield case
        if plan.stop_when_satisfied and tracker.satisfied:
            break


def generate_coverage_benchmark(
    spec: BenchmarkSpec,
    dataconfig=None,
    reference_set: ReferenceSet | None = None,
    plan: CoveragePlan | None = None,
    *,
    workers: int = 1,
) -> CoverageGenerationResult:
    """Materialize a coverage-driven benchmark and return its generation report."""

    dataconfig = dataconfig or dataconfig_by_name(spec.dataconfig_name)
    reference_set = reference_set or ReferenceSet.from_dataconfigs(dataconfig)
    plan = plan or coverage_plan_from_spec(spec, reference_set)
    tracker = CoverageTracker(plan)
    resolved_workers = _resolved_workers(workers)
    start = perf_counter()
    cases = list(
        stream_coverage_benchmark(
            spec,
            dataconfig=dataconfig,
            reference_set=reference_set,
            plan=plan,
            tracker=tracker,
            workers=resolved_workers,
        )
    )
    wall_time = perf_counter() - start
    report = tracker.to_dict()
    report["generation_profile"] = generation_run_report(
        mode="coverage_planned",
        n_cases=len(cases),
        wall_time_seconds=wall_time,
        workers=resolved_workers,
        generated_candidates=report["generated_cases"],
        accepted_cases=report["accepted_cases"],
    )
    return CoverageGenerationResult(cases=cases, report=report)
