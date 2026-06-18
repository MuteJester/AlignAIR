"""Generate benchmark cases from GenAIRR."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterator

from ...gym.crop import crop_record
from ...gym.curriculum import Curriculum
from ...gym.gym import build_experiment
from ...gym.targets import build_targets
from ...reference.reference_set import ReferenceSet
from ..core.schema import BenchmarkCase, BenchmarkSpec, GENES, GeneTruth, StratumSpec

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


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


def generate_benchmark(
    spec: BenchmarkSpec,
    dataconfig=None,
    reference_set: ReferenceSet | None = None,
) -> list[BenchmarkCase]:
    """Generate all cases described by ``spec``."""

    dataconfig = dataconfig or dataconfig_by_name(spec.dataconfig_name)
    reference_set = reference_set or ReferenceSet.from_dataconfigs(dataconfig)
    cases: list[BenchmarkCase] = []
    curriculum = Curriculum()
    for s_idx, stratum in enumerate(spec.strata):
        params = dict(curriculum.params(stratum.progress))
        params.update(stratum.param_overrides)
        exp = build_experiment(dataconfig, params)
        seed = spec.seed + stratum.seed_offset + 1009 * s_idx
        for i, record in enumerate(exp.stream_records(n=stratum.n, seed=seed)):
            if stratum.crop_to is not None:
                record = crop_record(record, stratum.crop_to)
            orientations = stratum.orientation_ids or (0,)
            orientation_id = orientations[i % len(orientations)]
            case_id = f"{spec.name}:{stratum.name}:{i:06d}"
            cases.append(_case_from_record(record, reference_set, stratum, case_id, orientation_id))
    return cases


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
            if stratum.crop_to is not None:
                record = crop_record(record, stratum.crop_to)
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
