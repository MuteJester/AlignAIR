"""Export benchmark inputs and reproducibility manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from ..core.schema import BenchmarkCase, BenchmarkSpec, GENES, ORIENTATION_NAMES
from ..generation import assess_benchmark_readiness, coverage_summary
from ...reference.reference_set import ReferenceSet


def _json_default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _sequence(case: BenchmarkCase, frame: str) -> str:
    if frame == "presented":
        return case.sequence
    if frame == "canonical":
        return case.canonical_sequence
    raise ValueError("frame must be 'presented' or 'canonical'")


def _wrap(text: str, width: int) -> Iterable[str]:
    if width <= 0:
        yield text
        return
    for start in range(0, len(text), width):
        yield text[start : start + width]


def save_fasta(
    cases: Iterable[BenchmarkCase],
    path: str | Path,
    *,
    frame: str = "presented",
    line_width: int = 80,
) -> None:
    """Write benchmark sequences as FASTA using ``case_id`` as the record id."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            orientation = ORIENTATION_NAMES.get(case.orientation_id, str(case.orientation_id))
            handle.write(
                f">{case.case_id} stratum={case.stratum} orientation={orientation} frame={frame}\n"
            )
            for line in _wrap(_sequence(case, frame), line_width):
                handle.write(f"{line}\n")


def airr_input_rows(
    cases: Iterable[BenchmarkCase],
    *,
    frame: str = "presented",
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Return AIRR-style input rows for running external tools."""

    rows = []
    for case in cases:
        row: dict[str, Any] = {
            "sequence_id": case.case_id,
            "sequence": _sequence(case, frame),
        }
        if include_metadata:
            row.update(
                {
                    "benchmark_case_id": case.case_id,
                    "benchmark_stratum": case.stratum,
                    "benchmark_frame": frame,
                    "benchmark_orientation_id": case.orientation_id,
                    "benchmark_orientation": ORIENTATION_NAMES.get(
                        case.orientation_id, str(case.orientation_id)
                    ),
                }
            )
            if case.record.get("locus"):
                row["locus"] = case.record["locus"]
        rows.append(row)
    return rows


def save_airr_input(
    cases: Iterable[BenchmarkCase],
    path: str | Path,
    *,
    frame: str = "presented",
    include_metadata: bool = False,
) -> None:
    """Write benchmark sequences as an AIRR-style TSV input table."""

    rows = airr_input_rows(cases, frame=frame, include_metadata=include_metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) or ["sequence_id", "sequence"]
    preferred = [name for name in ("sequence_id", "sequence") if name in fieldnames]
    fieldnames = preferred + [name for name in fieldnames if name not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def reference_set_summary(reference_set: ReferenceSet | None) -> dict[str, Any] | None:
    """Return allele counts and a stable content hash for a reference set."""

    if reference_set is None:
        return None
    digest = hashlib.sha256()
    genes: dict[str, Any] = {}
    for gene in sorted(reference_set.genes):
        ref = reference_set.gene(gene)
        digest.update(gene.encode("utf-8"))
        gene_digest = hashlib.sha256()
        for name, sequence in zip(ref.names, ref.sequences):
            payload = f"{name}\t{sequence}\n".encode("utf-8")
            digest.update(payload)
            gene_digest.update(payload)
        genes[gene.lower()] = {
            "n_alleles": len(ref.names),
            "sha256": gene_digest.hexdigest(),
        }
    return {
        "has_d": reference_set.has_d,
        "genes": genes,
        "sha256": digest.hexdigest(),
    }


def _truth_allele_counts(cases: list[BenchmarkCase]) -> dict[str, dict[str, int]]:
    counts = {gene: Counter() for gene in GENES}
    for case in cases:
        for gene, truth in case.genes.items():
            counts[gene].update(truth.calls)
    return {gene: dict(sorted(counter.items())) for gene, counter in counts.items()}


def build_benchmark_manifest(
    cases: Iterable[BenchmarkCase],
    *,
    spec: BenchmarkSpec | None = None,
    dataconfig_name: str | None = None,
    reference_set: ReferenceSet | None = None,
    generation_report: dict[str, Any] | None = None,
    frame: str = "presented",
    files: dict[str, str] | None = None,
    readiness_profile: str | None = "assay",
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for a generated benchmark."""

    case_list = list(cases)
    spec_dict = asdict(spec) if spec is not None else None
    resolved_dataconfig = dataconfig_name or (spec.dataconfig_name if spec is not None else None)
    manifest = {
        "manifest_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "software": {
            "alignair_version": _version("AlignAIR"),
            "genairr_version": _version("GenAIRR"),
            "python_version": platform.python_version(),
        },
        "benchmark": {
            "n_cases": len(case_list),
            "case_id_first": case_list[0].case_id if case_list else None,
            "case_id_last": case_list[-1].case_id if case_list else None,
            "strata": dict(sorted(Counter(case.stratum for case in case_list).items())),
            "input_frame": frame,
            "sequence_id_policy": "sequence_id equals benchmark case_id",
        },
        "generation": {
            "dataconfig_name": resolved_dataconfig,
            "spec": spec_dict,
            "generation_report": generation_report,
        },
        "reference": reference_set_summary(reference_set),
        "coverage": coverage_summary(case_list),
        "observed_truth": {
            "allele_counts": _truth_allele_counts(case_list),
        },
        "coordinate_conventions": {
            "case_jsonl": "0-based starts, end-exclusive sequence coordinates",
            "airr_prediction_inputs": "AIRR-style prediction starts are converted from 1-based by adapters",
        },
        "files": files or {},
    }
    if readiness_profile:
        manifest["readiness"] = assess_benchmark_readiness(
            case_list,
            reference_set=reference_set,
            profile=readiness_profile,
        )
    return manifest


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write a benchmark manifest as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def export_benchmark_inputs(
    cases: Iterable[BenchmarkCase],
    out_dir: str | Path,
    *,
    prefix: str = "benchmark",
    frame: str = "presented",
    include_airr_metadata: bool = False,
    spec: BenchmarkSpec | None = None,
    dataconfig_name: str | None = None,
    reference_set: ReferenceSet | None = None,
    generation_report: dict[str, Any] | None = None,
    readiness_profile: str | None = "assay",
) -> dict[str, str]:
    """Write FASTA, AIRR-input TSV, and manifest files for a benchmark."""

    case_list = list(cases)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "fasta": str(out_dir / f"{prefix}.fasta"),
        "airr_input_tsv": str(out_dir / f"{prefix}_airr_input.tsv"),
        "manifest": str(out_dir / f"{prefix}_manifest.json"),
    }
    save_fasta(case_list, files["fasta"], frame=frame)
    save_airr_input(
        case_list,
        files["airr_input_tsv"],
        frame=frame,
        include_metadata=include_airr_metadata,
    )
    manifest = build_benchmark_manifest(
        case_list,
        spec=spec,
        dataconfig_name=dataconfig_name,
        reference_set=reference_set,
        generation_report=generation_report,
        frame=frame,
        files=files,
        readiness_profile=readiness_profile,
    )
    save_manifest(manifest, files["manifest"])
    return files
