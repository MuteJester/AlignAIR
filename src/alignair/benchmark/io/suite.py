"""Export composed benchmark suite packs."""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from ..core.artifacts import (
    BENCHMARK_SUITE_MANIFEST,
    CURRENT_SCHEMA_VERSION,
    artifact_metadata,
)
from ..generation import coverage_summary, dataconfig_by_name, measurement_coverage_summary
from ..generation.suite import BenchmarkSuiteResult, SuiteComponentResult
from ..generation.suite.spec import BenchmarkSuiteSpec
from ...reference.reference_set import ReferenceSet
from .export import export_benchmark_inputs, reference_set_summary
from .jsonl import save_jsonl


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


def _reference_set_for_suite(suite: BenchmarkSuiteSpec) -> ReferenceSet:
    dataconfigs = [dataconfig_by_name(spec.dataconfig_name) for spec in suite.specs]
    return ReferenceSet.from_dataconfigs(*dataconfigs)


def _dataconfig_name_for_component(component: SuiteComponentResult) -> str:
    names = tuple(dict.fromkeys(spec.dataconfig_name for spec in component.component.specs))
    return ",".join(names)


def _reference_set_for_component(component: SuiteComponentResult) -> ReferenceSet:
    dataconfigs = [dataconfig_by_name(spec.dataconfig_name) for spec in component.component.specs]
    return ReferenceSet.from_dataconfigs(*dataconfigs)


def build_benchmark_suite_manifest(
    result: BenchmarkSuiteResult,
    *,
    reference_set: ReferenceSet | None = None,
    frame: str = "presented",
    files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for a composed benchmark suite."""

    case_list = list(result.cases)
    component_files = (files or {}).get("components", {})
    components = []
    for component in result.components:
        components.append(
            {
                "name": component.component.name,
                "role": component.component.role,
                "measurement_focus": component.component.measurement_focus,
                "readiness_profile": component.component.readiness_profile,
                "n_cases": len(component.cases),
                "spec_names": tuple(spec.name for spec in component.component.specs),
                "dataconfig_names": tuple(
                    dict.fromkeys(spec.dataconfig_name for spec in component.component.specs)
                ),
                "coverage": coverage_summary(component.cases),
                "measurement_coverage": measurement_coverage_summary(component.cases),
                "files": component_files.get(component.component.name, {}),
            }
        )

    return {
        "artifact": artifact_metadata(BENCHMARK_SUITE_MANIFEST),
        "manifest_version": CURRENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "software": {
            "alignair_version": _version("AlignAIR"),
            "genairr_version": _version("GenAIRR"),
            "python_version": platform.python_version(),
        },
        "suite": result.suite.to_dict(),
        "benchmark": {
            "n_cases": len(case_list),
            "input_frame": frame,
            "sequence_id_policy": "sequence_id equals benchmark case_id",
        },
        "generation": result.report.get("generation_report", {}),
        "reference": reference_set_summary(reference_set),
        "coverage": coverage_summary(case_list),
        "measurement_coverage": measurement_coverage_summary(case_list),
        "suite_readiness": result.report.get("suite_readiness", {}),
        "components": components,
        "files": files or {},
    }


def save_benchmark_suite_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write a benchmark suite manifest as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def export_benchmark_suite(
    result: BenchmarkSuiteResult,
    out_dir: str | Path,
    *,
    prefix: str = "benchmark_suite",
    frame: str = "presented",
    include_airr_metadata: bool = True,
    readiness_profile: str | None = "assay",
) -> dict[str, Any]:
    """Write combined and per-component artifacts for a generated benchmark suite."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_set = _reference_set_for_suite(result.suite)
    files: dict[str, Any] = {
        "cases_jsonl": str(out_dir / f"{prefix}.jsonl"),
        "components": {},
    }
    save_jsonl(result.cases, files["cases_jsonl"])
    files["combined"] = export_benchmark_inputs(
        result.cases,
        out_dir,
        prefix=f"{prefix}_combined",
        frame=frame,
        include_airr_metadata=include_airr_metadata,
        spec=None,
        dataconfig_name=",".join(dict.fromkeys(spec.dataconfig_name for spec in result.suite.specs)),
        reference_set=reference_set,
        generation_report=result.report,
        readiness_profile=readiness_profile,
    )

    for component in result.components:
        component_dir = out_dir / "components" / component.component.name
        component_spec = component.component.specs[0] if len(component.component.specs) == 1 else None
        files["components"][component.component.name] = export_benchmark_inputs(
            component.cases,
            component_dir,
            prefix=component.component.name,
            frame=frame,
            include_airr_metadata=include_airr_metadata,
            spec=component_spec,
            dataconfig_name=_dataconfig_name_for_component(component),
            reference_set=_reference_set_for_component(component),
            generation_report={
                "suite": result.suite.to_dict(),
                "component": component.component.to_dict(),
                "spec_reports": component.spec_reports,
            },
            readiness_profile=component.component.readiness_profile,
        )

    files["suite_manifest"] = str(out_dir / f"{prefix}_manifest.json")
    manifest = build_benchmark_suite_manifest(
        result,
        reference_set=reference_set,
        frame=frame,
        files=files,
    )
    save_benchmark_suite_manifest(manifest, files["suite_manifest"])
    return files
