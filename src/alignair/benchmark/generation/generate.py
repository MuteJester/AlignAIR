"""Generate benchmark cases from GenAIRR."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

from ...gym.crop import crop_record, crop_one_sided, anchor_c0
from ...gym.curriculum import Curriculum
from ...gym.gym import build_experiment
from ...gym.targets import build_targets
from ...reference.reference_set import ReferenceSet
from ..core.schema import BenchmarkCase, BenchmarkSpec, GENES, GeneTruth, StratumSpec

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _apply_crop(record, stratum):
    """One-sided germline/J anchor (adaptive) takes precedence over the symmetric crop_to."""
    if getattr(stratum, "anchor", None) is not None:
        return crop_one_sided(record, anchor_c0(record, stratum.anchor))
    if stratum.crop_to is not None:
        return crop_record(record, stratum.crop_to)
    return record


@dataclass
class BenchmarkGenerationResult:
    """Materialized fixed-size benchmark plus generation runtime report."""

    cases: list[BenchmarkCase]
    report: dict[str, Any]


def _orient_sequence(seq: str, orientation_id: int) -> str:
    if orientation_id == 1:
        return seq.translate(_COMP)[::-1].upper()
    if orientation_id == 2:
        return seq.translate(_COMP).upper()
    if orientation_id == 3:
        return seq[::-1].upper()
    return seq.upper()


def _orient_labels(labels: list[int], orientation_id: int) -> list[int]:
    return list(reversed(labels)) if orientation_id in (1, 3) else list(labels)


def _presented_gene_truth(truth: GeneTruth, length: int, orientation_id: int) -> GeneTruth:
    if orientation_id not in (1, 3) or truth.sequence_start is None or truth.sequence_end is None:
        return truth
    return GeneTruth(
        calls=truth.calls,
        primary=truth.primary,
        sequence_start=length - truth.sequence_end,
        sequence_end=length - truth.sequence_start,
        germline_start=truth.germline_start,
        germline_end=truth.germline_end,
    )


def _gene_truths(target: dict[str, Any]) -> dict[str, GeneTruth]:
    out = {}
    for g in GENES:
        G = g.upper()
        calls = tuple(sorted(target["calls"].get(G, ())))
        primary = target.get("primary", {}).get(G)
        ss, se = target.get("inseq", {}).get(g, (None, None))
        gs, ge = target.get("germline", {}).get(g, (None, None))
        out[g] = GeneTruth(
            calls=calls,
            primary=primary,
            sequence_start=ss,
            sequence_end=se,
            germline_start=gs,
            germline_end=ge,
        )
    return out


def _record_tags(record: dict[str, Any], stratum: StratumSpec, orientation_id: int) -> dict[str, Any]:
    seq = str(record["sequence"])
    return {
        "stratum_tags": list(stratum.tags),
        "description": stratum.description,
        "length": len(seq),
        "orientation_id": orientation_id,
        "mutation_rate": float(record.get("mutation_rate", 0.0) or 0.0),
        "n_indels": int(record.get("n_indels", 0) or 0),
        "n_quality_errors": int(record.get("n_quality_errors", 0) or 0),
        "n_pcr_errors": int(record.get("n_pcr_errors", 0) or 0),
        "productive": bool(record.get("productive", False)),
        "d_inverted": bool(record.get("d_inverted", False)),
        "crop_to": stratum.crop_to,
    }


def _case_from_record(
    record: dict[str, Any],
    reference_set: ReferenceSet,
    stratum: StratumSpec,
    case_id: str,
    orientation_id: int,
) -> BenchmarkCase:
    has_d = reference_set.has_d
    target = build_targets(record, reference_set, has_d=has_d)
    canonical_sequence = str(record["sequence"]).upper()
    sequence = _orient_sequence(canonical_sequence, orientation_id)
    genes = _gene_truths(target)
    presented_genes = {
        g: _presented_gene_truth(t, len(canonical_sequence), orientation_id) for g, t in genes.items()
    }
    region_labels = [int(x) for x in target["region_labels"].tolist()]
    state_labels = [int(x) for x in target["state_labels"].tolist()]
    scalars = {
        "noise_count": float(target["noise_count"]),
        "mutation_rate": float(target["mutation_rate"]),
        "indel_count": float(target["indel_count"]),
        "productive": float(target["productive"]),
    }
    return BenchmarkCase(
        case_id=case_id,
        stratum=stratum.name,
        sequence=sequence,
        canonical_sequence=canonical_sequence,
        orientation_id=orientation_id,
        genes=genes,
        presented_genes=presented_genes,
        region_labels=region_labels,
        state_labels=state_labels,
        presented_region_labels=_orient_labels(region_labels, orientation_id),
        presented_state_labels=_orient_labels(state_labels, orientation_id),
        scalars=scalars,
        tags=_record_tags(record, stratum, orientation_id),
        record=dict(record),
    )


def dataconfig_by_name(name: str):
    """Resolve a GenAIRR data config by name from ``GenAIRR.data``."""

    import GenAIRR.data as gdata

    return getattr(gdata, name)


def _resolved_workers(workers: int | None) -> int:
    try:
        resolved = int(workers or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("workers must be a positive integer") from exc
    if resolved < 1:
        raise ValueError("workers must be a positive integer")
    return resolved


def generation_run_report(
    *,
    mode: str,
    n_cases: int,
    wall_time_seconds: float,
    workers: int = 1,
    generated_candidates: int | None = None,
    accepted_cases: int | None = None,
) -> dict[str, Any]:
    """Return normalized runtime stats for a benchmark generation run."""

    generated = int(generated_candidates if generated_candidates is not None else n_cases)
    accepted = int(accepted_cases if accepted_cases is not None else n_cases)
    wall_time = float(wall_time_seconds)
    return {
        "mode": mode,
        "workers": int(workers),
        "parallel": int(workers) > 1,
        "wall_time_seconds": wall_time,
        "n_cases": int(n_cases),
        "generated_candidates": generated,
        "accepted_cases": accepted,
        "acceptance_rate": (accepted / generated) if generated else None,
        "cases_per_second": (n_cases / wall_time) if wall_time > 0 else None,
        "candidates_per_second": (generated / wall_time) if wall_time > 0 else None,
    }


def _stratum_cases(
    spec: BenchmarkSpec,
    dataconfig,
    reference_set: ReferenceSet,
    s_idx: int,
    stratum: StratumSpec,
) -> list[BenchmarkCase]:
    curriculum = Curriculum()
    params = dict(curriculum.params(stratum.progress))
    params.update(stratum.param_overrides)
    exp = build_experiment(dataconfig, params)
    seed = spec.seed + stratum.seed_offset + 1009 * s_idx
    cases: list[BenchmarkCase] = []
    for i, record in enumerate(exp.stream_records(n=stratum.n, seed=seed)):
        record = _apply_crop(record, stratum)
        orientations = stratum.orientation_ids or (0,)
        orientation_id = orientations[i % len(orientations)]
        case_id = f"{spec.name}:{stratum.name}:{i:06d}"
        cases.append(_case_from_record(record, reference_set, stratum, case_id, orientation_id))
    return cases


def _stratum_worker(task: tuple[BenchmarkSpec, int, StratumSpec, ReferenceSet | None]) -> tuple[int, list[BenchmarkCase]]:
    spec, s_idx, stratum, reference_set = task
    dataconfig = dataconfig_by_name(spec.dataconfig_name)
    resolved_reference_set = reference_set or ReferenceSet.from_dataconfigs(dataconfig)
    return s_idx, _stratum_cases(spec, dataconfig, resolved_reference_set, s_idx, stratum)


def candidate_round_cases(
    spec: BenchmarkSpec,
    dataconfig,
    reference_set: ReferenceSet,
    s_idx: int,
    stratum: StratumSpec,
    round_idx: int,
) -> list[BenchmarkCase]:
    """Generate one deterministic stratum/round candidate chunk for coverage planning."""

    curriculum = Curriculum()
    params = dict(curriculum.params(stratum.progress))
    params.update(stratum.param_overrides)
    exp = build_experiment(dataconfig, params)
    seed = spec.seed + stratum.seed_offset + 1009 * s_idx + 1_000_003 * round_idx
    cases: list[BenchmarkCase] = []
    for i, record in enumerate(exp.stream_records(n=stratum.n, seed=seed)):
        record = _apply_crop(record, stratum)
        orientations = stratum.orientation_ids or (0,)
        stratum_index = round_idx * stratum.n + i
        orientation_id = orientations[stratum_index % len(orientations)]
        if round_idx == 0:
            case_id = f"{spec.name}:{stratum.name}:{i:06d}"
        else:
            case_id = f"{spec.name}:{stratum.name}:extra{round_idx:03d}:{i:06d}"
        cases.append(_case_from_record(record, reference_set, stratum, case_id, orientation_id))
    return cases


def candidate_round_worker(
    task: tuple[BenchmarkSpec, int, StratumSpec, int, ReferenceSet | None],
) -> tuple[int, list[BenchmarkCase]]:
    """Worker entry point for deterministic coverage-planned candidate chunks."""

    spec, s_idx, stratum, round_idx, reference_set = task
    dataconfig = dataconfig_by_name(spec.dataconfig_name)
    resolved_reference_set = reference_set or ReferenceSet.from_dataconfigs(dataconfig)
    return s_idx, candidate_round_cases(
        spec,
        dataconfig,
        resolved_reference_set,
        s_idx,
        stratum,
        round_idx,
    )


def generate_benchmark(
    spec: BenchmarkSpec,
    dataconfig=None,
    reference_set: ReferenceSet | None = None,
    *,
    workers: int = 1,
) -> list[BenchmarkCase]:
    """Generate all cases described by ``spec``."""

    dataconfig = dataconfig or dataconfig_by_name(spec.dataconfig_name)
    reference_set = reference_set or ReferenceSet.from_dataconfigs(dataconfig)
    resolved_workers = _resolved_workers(workers)
    active = tuple((s_idx, stratum) for s_idx, stratum in enumerate(spec.strata) if stratum.n > 0)
    if resolved_workers == 1 or len(active) <= 1:
        cases: list[BenchmarkCase] = []
        for s_idx, stratum in active:
            cases.extend(_stratum_cases(spec, dataconfig, reference_set, s_idx, stratum))
        return cases

    tasks = [(spec, s_idx, stratum, reference_set) for s_idx, stratum in active]
    max_workers = min(resolved_workers, len(tasks))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_stratum_worker, tasks))
    cases = []
    for _, chunk in sorted(results, key=lambda item: item[0]):
        cases.extend(chunk)
    return cases


def generate_benchmark_with_report(
    spec: BenchmarkSpec,
    dataconfig=None,
    reference_set: ReferenceSet | None = None,
    *,
    workers: int = 1,
) -> BenchmarkGenerationResult:
    """Generate fixed-size benchmark cases and return generation runtime stats."""

    start = perf_counter()
    cases = generate_benchmark(
        spec,
        dataconfig=dataconfig,
        reference_set=reference_set,
        workers=workers,
    )
    wall_time = perf_counter() - start
    report = generation_run_report(
        mode="fixed_size",
        n_cases=len(cases),
        wall_time_seconds=wall_time,
        workers=_resolved_workers(workers),
        generated_candidates=len(cases),
        accepted_cases=len(cases),
    )
    return BenchmarkGenerationResult(cases=cases, report=report)


def stream_benchmark(
    spec: BenchmarkSpec,
    dataconfig=None,
    reference_set: ReferenceSet | None = None,
) -> Iterator[BenchmarkCase]:
    """Yield benchmark cases online without materializing the full dataset."""

    dataconfig = dataconfig or dataconfig_by_name(spec.dataconfig_name)
    reference_set = reference_set or ReferenceSet.from_dataconfigs(dataconfig)
    curriculum = Curriculum()
    for s_idx, stratum in enumerate(spec.strata):
        params = dict(curriculum.params(stratum.progress))
        params.update(stratum.param_overrides)
        exp = build_experiment(dataconfig, params)
        seed = spec.seed + stratum.seed_offset + 1009 * s_idx
        for i, record in enumerate(exp.stream_records(n=stratum.n, seed=seed)):
            record = _apply_crop(record, stratum)
            orientations = stratum.orientation_ids or (0,)
            orientation_id = orientations[i % len(orientations)]
            case_id = f"{spec.name}:{stratum.name}:{i:06d}"
            yield _case_from_record(record, reference_set, stratum, case_id, orientation_id)


def coverage_summary(cases: list[BenchmarkCase]) -> dict[str, Any]:
    """Summarize benchmark coverage by stratum, allele, ambiguity, length, and stressors."""

    by_stratum = Counter(c.stratum for c in cases)
    allele_counts: dict[str, Counter] = {g: Counter() for g in GENES}
    multi_counts = Counter()
    lengths = [len(c.sequence) for c in cases]
    orientation = Counter(c.orientation_id for c in cases)
    for case in cases:
        for g, truth in case.genes.items():
            if len(truth.calls) > 1:
                multi_counts[g] += 1
            for call in truth.calls:
                allele_counts[g][call] += 1
    allele_summary = {
        g: {
            "n_observed": len(counts),
            "min_count": min(counts.values()) if counts else 0,
            "max_count": max(counts.values()) if counts else 0,
            "mean_count": (sum(counts.values()) / len(counts)) if counts else 0.0,
        }
        for g, counts in allele_counts.items()
    }
    return {
        "n_cases": len(cases),
        "by_stratum": dict(by_stratum),
        "orientation": dict(orientation),
        "length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        },
        "alleles": allele_summary,
        "multi_call_cases": dict(multi_counts),
    }
