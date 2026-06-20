"""Bootstrap uncertainty estimates for GenAIRR-backed benchmark metrics."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

from ..core.schema import BenchmarkCase, GENES
from .metrics import score_cases


DEFAULT_BOOTSTRAP_METRICS: tuple[str, ...] = tuple(
    [f"genes.{gene}.{metric}" for gene in GENES for metric in (
        "call_top1_in_set",
        "call_set_f1",
        "ss_mae",
        "se_mae",
        "gs_mae",
        "ge_mae",
    )]
    + [
        "global.junction_nt_exact",
        "global.junction_aa_exact",
        "global.productive_acc",
        "global.orientation_acc",
        "global.required_field_presence",
        "global.parseable_airr_rate",
    ]
)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return values[lower]
    frac = pos - lower
    return values[lower] * (1.0 - frac) + values[upper] * frac


def _metric_value(scores: dict[str, Any], path: str) -> float | None:
    parts = path.split(".")
    if len(parts) == 2 and parts[0] == "global":
        return _finite(scores.get("global", {}).get(parts[1]))
    if len(parts) == 3 and parts[0] == "genes":
        return _finite(scores.get("genes", {}).get(parts[1], {}).get(parts[2]))
    return None


def _score_metric_values(
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
    *,
    frame: str,
    metric_paths: tuple[str, ...],
) -> dict[str, float]:
    scores = score_cases(
        cases,
        predictions,
        frame=frame,
        include_strata=False,
        include_expensive_record_fields=False,
    )
    values = {}
    for path in metric_paths:
        value = _metric_value(scores, path)
        if value is not None:
            values[path] = value
    return values


def _intervals_from_samples(
    point_values: dict[str, float],
    samples: dict[str, list[float]],
    *,
    confidence: float,
    n_cases: int,
) -> dict[str, dict[str, Any]]:
    alpha = 1.0 - confidence
    out = {}
    for path, point in sorted(point_values.items()):
        values = samples.get(path, [])
        if not values:
            continue
        out[path] = {
            "point": point,
            "mean": sum(values) / len(values),
            "ci_low": _quantile(values, alpha / 2.0),
            "ci_high": _quantile(values, 1.0 - alpha / 2.0),
            "n_bootstrap": len(values),
            "n_cases": n_cases,
        }
    return out


def _bootstrap_scope(
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
    *,
    frame: str,
    metric_paths: tuple[str, ...],
    n_bootstrap: int,
    confidence: float,
    rng: random.Random,
) -> dict[str, dict[str, Any]]:
    if not cases:
        return {}
    point_values = _score_metric_values(cases, predictions, frame=frame, metric_paths=metric_paths)
    samples: dict[str, list[float]] = defaultdict(list)
    n_cases = len(cases)
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n_cases) for _ in range(n_cases)]
        sample_cases = [cases[i] for i in indices]
        sample_predictions = [predictions[i] for i in indices]
        values = _score_metric_values(
            sample_cases,
            sample_predictions,
            frame=frame,
            metric_paths=metric_paths,
        )
        for path, value in values.items():
            samples[path].append(value)
    return _intervals_from_samples(point_values, samples, confidence=confidence, n_cases=n_cases)


def bootstrap_metric_intervals(
    cases: Iterable[BenchmarkCase],
    predictions: Iterable[dict[str, Any] | None],
    *,
    frame: str = "canonical",
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 123,
    metric_paths: Iterable[str] | None = None,
    include_strata: bool = True,
    min_stratum_cases: int = 2,
) -> dict[str, Any]:
    """Estimate confidence intervals by paired case-level bootstrap.

    This resamples benchmark case/prediction pairs. It does not create new truth
    labels; all scored quantities remain derived from GenAIRR benchmark cases.
    """

    case_list = list(cases)
    prediction_list = list(predictions)
    if len(case_list) != len(prediction_list):
        raise ValueError(f"case/prediction length mismatch: {len(case_list)} != {len(prediction_list)}")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    paths = tuple(metric_paths or DEFAULT_BOOTSTRAP_METRICS)
    rng = random.Random(seed)
    out: dict[str, Any] = {
        "method": "paired_case_bootstrap",
        "truth_source": "GenAIRR benchmark cases",
        "n_cases": len(case_list),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "seed": seed,
        "frame": frame,
        "metric_paths": list(paths),
        "overall": _bootstrap_scope(
            case_list,
            prediction_list,
            frame=frame,
            metric_paths=paths,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            rng=rng,
        ),
        "by_stratum": {},
        "skipped_strata": [],
    }
    if include_strata:
        by_stratum: dict[str, list[int]] = defaultdict(list)
        for idx, case in enumerate(case_list):
            by_stratum[case.stratum].append(idx)
        for stratum, indices in sorted(by_stratum.items()):
            if len(indices) < min_stratum_cases:
                out["skipped_strata"].append(
                    {
                        "stratum": stratum,
                        "n_cases": len(indices),
                        "reason": f"requires at least {min_stratum_cases} cases",
                    }
                )
                continue
            stratum_cases = [case_list[i] for i in indices]
            stratum_predictions = [prediction_list[i] for i in indices]
            out["by_stratum"][stratum] = _bootstrap_scope(
                stratum_cases,
                stratum_predictions,
                frame=frame,
                metric_paths=paths,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                rng=rng,
            )
    return out
