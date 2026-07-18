from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ...core import metric_registry

_METRIC_SPEC_REQUIRED_FIELDS = (
    "key",
    "higher_is_better",
    "pass_threshold",
    "warn_threshold",
)
_SCORING_MANIFEST_REQUIRED_FIELDS = (
    "name",
    "scope",
    "metric_keys",
    "aggregation",
)


def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def _observed_metric_keys(report: Mapping[str, Any]) -> set[str]:
    results = report.get("results")
    if not isinstance(results, Mapping):
        return set()
    overall = results.get("overall")
    if not isinstance(overall, Mapping):
        return set()

    global_metrics = overall.get("global") or {}
    keys = set()
    if isinstance(global_metrics, Mapping):
        keys.update(str(key) for key, value in global_metrics.items() if _is_finite(value))
    genes = overall.get("genes") or {}
    if isinstance(genes, Mapping):
        for metrics in genes.values():
            if not isinstance(metrics, Mapping):
                continue
            keys.update(str(key) for key, value in metrics.items() if _is_finite(value))
    return keys


def _criteria_metric_keys(report: Mapping[str, Any]) -> set[str]:
    criteria = report.get("criteria") or ()
    if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
        return set()
    keys = set()
    for criterion in criteria:
        if not isinstance(criterion, Mapping):
            continue
        for key in criterion.get("metric_keys") or ():
            keys.add(str(key))
    return keys


def _embedded_metric_registry(report: Mapping[str, Any]) -> tuple[set[str], tuple[str, ...], tuple[str, ...]]:
    catalog = report.get("metric_registry")
    if catalog is None:
        return set(metric_registry()), ("missing_metric_registry",), ()
    if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes)):
        return set(), ("malformed_metric_registry",), ()

    keys = []
    malformed = []
    for idx, row in enumerate(catalog):
        if not isinstance(row, Mapping):
            malformed.append(f"metric_registry[{idx}]")
            continue
        key = row.get("key")
        if not isinstance(key, str) or not key:
            malformed.append(f"metric_registry[{idx}].key")
            continue
        missing = tuple(field for field in _METRIC_SPEC_REQUIRED_FIELDS if field not in row)
        if missing:
            malformed.append(f"{key}:{','.join(missing)}")
        keys.append(key)

    duplicates = tuple(sorted(key for key, count in Counter(keys).items() if count > 1))
    errors = []
    if malformed:
        errors.append("malformed_metric_registry")
    if duplicates:
        errors.append("duplicate_metric_registry_keys")
    return set(keys), tuple(errors), duplicates


def _embedded_scoring_manifest(
    report: Mapping[str, Any],
    *,
    registry_metric_keys: set[str],
) -> tuple[set[str], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    catalog = report.get("scoring_manifest")
    if catalog is None:
        return set(), (), (), ()
    if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes)):
        return set(), ("malformed_scoring_manifest",), (), ()

    names = []
    metric_keys = []
    malformed = []
    for idx, row in enumerate(catalog):
        if not isinstance(row, Mapping):
            malformed.append(f"scoring_manifest[{idx}]")
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name:
            malformed.append(f"scoring_manifest[{idx}].name")
            continue
        missing = tuple(field for field in _SCORING_MANIFEST_REQUIRED_FIELDS if field not in row)
        if missing:
            malformed.append(f"{name}:{','.join(missing)}")
        names.append(name)
        keys = row.get("metric_keys") or ()
        if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
            malformed.append(f"{name}:metric_keys")
            continue
        metric_keys.extend(str(key) for key in keys if isinstance(key, str) and key)

    duplicate_names = tuple(sorted(name for name, count in Counter(names).items() if count > 1))
    without_registry = tuple(sorted(set(metric_keys) - registry_metric_keys))
    errors = []
    if malformed:
        errors.append("malformed_scoring_manifest")
    if duplicate_names:
        errors.append("duplicate_scoring_component_names")
    if without_registry:
        errors.append("scoring_manifest_metric_keys_without_registry")
    return set(metric_keys), tuple(errors), duplicate_names, without_registry
