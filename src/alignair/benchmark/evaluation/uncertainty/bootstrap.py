from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from ...core.schema import BenchmarkCase
from .constants import DEFAULT_BOOTSTRAP_METRICS
from .math_utils import _quantile
from .metrics import _score_metric_values


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
