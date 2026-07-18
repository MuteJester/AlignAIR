"""Runtime audit of scorer manifests against observed benchmark metrics."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..audit import case_truth_value
from ...core.schema import BenchmarkCase, GENES
from .manifest import SCORING_MANIFESTS, ScoringComponentManifest, scoring_manifest_catalog


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _extract_overall(scores_or_report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    payload = scores_or_report or {}
    if "results" in payload:
        results = payload.get("results") or {}
        return results.get("overall") or {}
    return payload


def _observed_metric_keys(scores_or_report: Mapping[str, Any] | None) -> dict[str, Any]:
    overall = _extract_overall(scores_or_report)
    global_keys = {
        str(key)
        for key, value in (overall.get("global") or {}).items()
        if _finite(value) is not None
    }
    gene_keys: dict[str, set[str]] = {}
    for gene, metrics in (overall.get("genes") or {}).items():
        if not isinstance(metrics, Mapping):
            continue
        gene_keys[str(gene)] = {
            str(key)
            for key, value in metrics.items()
            if _finite(value) is not None
        }
    combined = set(global_keys)
    for keys in gene_keys.values():
        combined.update(keys)
    return {
        "global": global_keys,
        "genes": gene_keys,
        "all": combined,
    }


def _expand_gene_calls() -> tuple[str, ...]:
    return tuple(field for gene in GENES for field in (f"{gene}_call", f"{gene}_calls"))


def _expand_gene_coordinates() -> tuple[str, ...]:
    return tuple(
        f"{gene}_{kind}_{side}"
        for gene in GENES
        for kind in ("sequence", "germline")
        for side in ("start", "end")
    )


def _expand_gene_cigars() -> tuple[str, ...]:
    return tuple(f"{gene}_cigar" for gene in GENES)


def _expand_requirement(requirement: str) -> tuple[str, ...]:
    special = {
        "v/d/j calls": _expand_gene_calls(),
        "v/d/j sequence/germline coordinates": _expand_gene_coordinates(),
        "v/d/j coordinates": _expand_gene_coordinates(),
        "v/d/j cigar": _expand_gene_cigars(),
    }
    if requirement in special:
        return special[requirement]
    if requirement.endswith("_start/end"):
        stem = requirement[: -len("_start/end")]
        return (f"{stem}_start", f"{stem}_end")
    if "/" in requirement and " " not in requirement:
        return tuple(part for part in requirement.split("/") if part)
    if " " in requirement:
        return ()
    return (requirement,)


def _prediction_requirement_availability(
    predictions: Iterable[Mapping[str, Any] | None] | None,
    requirements: Iterable[str],
) -> dict[str, dict[str, Any]]:
    prediction_list = list(predictions or [])
    out: dict[str, dict[str, Any]] = {}
    for requirement in sorted(set(requirements)):
        field_keys = _expand_requirement(requirement)
        if not prediction_list:
            out[requirement] = {
                "field_keys": field_keys,
                "machine_checkable": bool(field_keys),
                "n_present_any": None,
                "n_predictions": 0,
                "fraction_present_any": None,
            }
            continue
        if not field_keys:
            out[requirement] = {
                "field_keys": (),
                "machine_checkable": False,
                "n_present_any": None,
                "n_predictions": len(prediction_list),
                "fraction_present_any": None,
            }
            continue
        present = 0
        for prediction in prediction_list:
            pred = prediction or {}
            if any(key in pred and not _is_missing(pred.get(key)) for key in field_keys):
                present += 1
        out[requirement] = {
            "field_keys": field_keys,
            "machine_checkable": True,
            "n_present_any": present,
            "n_predictions": len(prediction_list),
            "fraction_present_any": present / len(prediction_list),
        }
    return out


def _truth_requirement_availability(
    cases: Iterable[BenchmarkCase] | None,
    requirements: Iterable[str],
) -> dict[str, dict[str, Any]]:
    case_list = list(cases or [])
    out: dict[str, dict[str, Any]] = {}
    for requirement in sorted(set(requirements)):
        field_keys = _expand_requirement(requirement)
        if not case_list:
            out[requirement] = {
                "field_keys": field_keys,
                "machine_checkable": bool(field_keys),
                "n_present_any": None,
                "n_cases": 0,
                "fraction_present_any": None,
            }
            continue
        if not field_keys:
            out[requirement] = {
                "field_keys": (),
                "machine_checkable": False,
                "n_present_any": None,
                "n_cases": len(case_list),
                "fraction_present_any": None,
            }
            continue
        present = 0
        for case in case_list:
            if any(not _is_missing(case_truth_value(case, key)) for key in field_keys):
                present += 1
        out[requirement] = {
            "field_keys": field_keys,
            "machine_checkable": True,
            "n_present_any": present,
            "n_cases": len(case_list),
            "fraction_present_any": present / len(case_list),
        }
    return out


def _manifest_rows(report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if report and isinstance(report.get("scoring_manifest"), Sequence) and not isinstance(
        report.get("scoring_manifest"), (str, bytes)
    ):
        return [dict(row) for row in report["scoring_manifest"] if isinstance(row, Mapping)]
    return scoring_manifest_catalog(include_metric_specs=False)


def _declared_scope_metric_keys(rows: Sequence[Mapping[str, Any]], scope: str) -> set[str]:
    return {
        str(key)
        for row in rows
        if row.get("scope") == scope
        for key in row.get("metric_keys") or ()
    }


def _component_row(
    manifest: Mapping[str, Any],
    *,
    observed: dict[str, Any],
    cases: Iterable[BenchmarkCase] | None,
    predictions: Iterable[Mapping[str, Any] | None] | None,
) -> dict[str, Any]:
    name = str(manifest.get("name"))
    scope = str(manifest.get("scope"))
    metric_keys = tuple(str(key) for key in manifest.get("metric_keys") or ())
    if scope == "global":
        observed_keys = tuple(key for key in metric_keys if key in observed["global"])
        observed_by_gene: dict[str, tuple[str, ...]] = {}
    elif scope == "gene":
        observed_union = set().union(*observed["genes"].values()) if observed["genes"] else set()
        observed_keys = tuple(key for key in metric_keys if key in observed_union)
        observed_by_gene = {
            gene: tuple(key for key in metric_keys if key in keys)
            for gene, keys in sorted(observed["genes"].items())
        }
    else:
        observed_keys = tuple(key for key in metric_keys if key in observed["all"])
        observed_by_gene = {}

    missing_keys = tuple(key for key in metric_keys if key not in set(observed_keys))
    if not observed_keys:
        status = "not_exercised"
    elif missing_keys:
        status = "partially_exercised"
    else:
        status = "fully_exercised"

    return {
        "name": name,
        "scope": scope,
        "status": status,
        "declared_metric_keys": metric_keys,
        "observed_metric_keys": observed_keys,
        "missing_metric_keys": missing_keys,
        "metric_coverage_fraction": len(observed_keys) / len(metric_keys) if metric_keys else 0.0,
        "observed_by_gene": observed_by_gene,
        "prediction_field_availability": _prediction_requirement_availability(
            predictions,
            manifest.get("required_prediction_fields") or (),
        ),
        "truth_field_availability": _truth_requirement_availability(
            cases,
            manifest.get("ground_truth_fields") or (),
        ),
        "scenario_axes": tuple(manifest.get("scenario_axes") or ()),
        "aggregation": manifest.get("aggregation"),
        "description": manifest.get("description"),
    }


def audit_scoring_runtime(
    scores_or_report: Mapping[str, Any] | None = None,
    *,
    cases: Iterable[BenchmarkCase] | None = None,
    predictions: Iterable[Mapping[str, Any] | None] | None = None,
    manifests: tuple[ScoringComponentManifest, ...] = SCORING_MANIFESTS,
) -> dict[str, Any]:
    """Audit declared scorer outputs against metrics observed in this run."""

    if manifests is SCORING_MANIFESTS:
        manifest_rows = _manifest_rows(scores_or_report)
    else:
        manifest_rows = [manifest.to_dict(include_metric_specs=False) for manifest in manifests]
    observed = _observed_metric_keys(scores_or_report)
    declared_global = _declared_scope_metric_keys(manifest_rows, "global")
    declared_gene = _declared_scope_metric_keys(manifest_rows, "gene")
    declared_all = {
        str(key)
        for manifest in manifest_rows
        for key in manifest.get("metric_keys") or ()
    }
    observed_global_without_manifest = observed["global"] - declared_global
    observed_gene_without_manifest = (
        set().union(*observed["genes"].values()) - declared_gene if observed["genes"] else set()
    )
    component_rows = [
        _component_row(row, observed=observed, cases=cases, predictions=predictions)
        for row in manifest_rows
    ]
    status_counts = {
        status: sum(1 for row in component_rows if row["status"] == status)
        for status in ("fully_exercised", "partially_exercised", "not_exercised")
    }
    observed_declared = observed["all"] & declared_all
    return {
        "summary": {
            "n_components": len(component_rows),
            "component_status_counts": status_counts,
            "n_declared_metric_keys": len(declared_all),
            "n_observed_metric_keys": len(observed["all"]),
            "n_observed_declared_metric_keys": len(observed_declared),
            "n_observed_metric_keys_without_manifest": len(observed["all"] - declared_all),
            "n_global_metric_keys_without_global_manifest": len(observed_global_without_manifest),
            "n_gene_metric_keys_without_gene_manifest": len(observed_gene_without_manifest),
            "has_case_truth_audit": cases is not None,
            "has_prediction_field_audit": predictions is not None,
        },
        "observed_metric_keys": {
            "global": tuple(sorted(observed["global"])),
            "genes": {gene: tuple(sorted(keys)) for gene, keys in sorted(observed["genes"].items())},
            "all": tuple(sorted(observed["all"])),
        },
        "declared_metric_keys": {
            "global": tuple(sorted(declared_global)),
            "gene": tuple(sorted(declared_gene)),
            "all": tuple(sorted(declared_all)),
        },
        "observed_metric_keys_without_manifest": tuple(sorted(observed["all"] - declared_all)),
        "global_metric_keys_without_global_manifest": tuple(sorted(observed_global_without_manifest)),
        "gene_metric_keys_without_gene_manifest": tuple(sorted(observed_gene_without_manifest)),
        "components": component_rows,
    }
