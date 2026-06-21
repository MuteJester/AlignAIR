"""Paired model-vs-model benchmark comparison reports."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

from ..core.schema import BenchmarkCase
from .matching import align_predictions_to_cases
from .metrics import score_one_case
from .uncertainty import DEFAULT_BOOTSTRAP_METRICS

DEFAULT_COMPARISON_METRICS: tuple[str, ...] = DEFAULT_BOOTSTRAP_METRICS


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


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _direction_for_metric(path: str, overrides: dict[str, str] | None = None) -> tuple[int, str]:
    if overrides and path in overrides:
        value = overrides[path].strip().lower()
        if value in {"higher", "higher_is_better", "max"}:
            return 1, "higher_is_better"
        if value in {"lower", "lower_is_better", "min"}:
            return -1, "lower_is_better"
        raise ValueError(f"unsupported metric direction for {path}: {overrides[path]}")

    metric = path.split(".")[-1]
    lower_is_better = {
        "missing_call_rate",
        "missing_coordinate_rate",
        "coordinate_frame_error_rate",
        "off_by_one_rate",
        "negative_span_rate",
        "overlap_rate",
        "outside_genotype_call_rate",
        "overcall_rate",
        "undercall_rate",
        "false_positive_alignment_rate",
        "false_shm_from_noise_rate",
        "set_size_mae",
    }
    if metric.endswith(("_mae", "_edit_distance", "_error_rate")) or metric in lower_is_better:
        return -1, "lower_is_better"
    return 1, "higher_is_better"


def _metric_verdict(
    advantage: float | None,
    *,
    model_a_name: str,
    model_b_name: str,
    practical_delta: float,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> tuple[str, str | None]:
    if advantage is None:
        return "not_scored", None

    if ci_low is not None and ci_high is not None:
        if ci_low > practical_delta:
            return "model_b_better", model_b_name
        if ci_high < -practical_delta:
            return "model_a_better", model_a_name
        if ci_low >= -practical_delta and ci_high <= practical_delta:
            return "tie_or_negligible", None
        return "inconclusive", None

    if advantage > practical_delta:
        return "model_b_better", model_b_name
    if advantage < -practical_delta:
        return "model_a_better", model_a_name
    return "tie_or_negligible", None


def _overall_verdict(counts: dict[str, int]) -> str:
    if counts.get("not_scored", 0) and sum(v for k, v in counts.items() if k != "not_scored") == 0:
        return "not_scored"
    a = counts.get("model_a_better", 0)
    b = counts.get("model_b_better", 0)
    ties = counts.get("tie_or_negligible", 0)
    inconclusive = counts.get("inconclusive", 0)
    if a and b:
        return "mixed"
    if b and not a:
        return "model_b_better"
    if a and not b:
        return "model_a_better"
    if ties and not inconclusive:
        return "tie_or_negligible"
    return "inconclusive"


def _summarize_rows(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows.values():
        counts[row["verdict"]] += 1
    return {
        "n_metrics": len(rows),
        "n_scored_metrics": sum(1 for row in rows.values() if row["verdict"] != "not_scored"),
        "verdict": _overall_verdict(counts),
        "verdict_counts": dict(sorted(counts.items())),
    }


def _bootstrap_metric(
    pairs: list[tuple[float, float]],
    *,
    direction: int,
    n_bootstrap: int,
    confidence: float,
    practical_delta: float,
    rng: random.Random,
) -> dict[str, Any]:
    if n_bootstrap <= 0 or not pairs:
        return {}

    raw_samples: list[float] = []
    advantage_samples: list[float] = []
    n = len(pairs)
    for _ in range(n_bootstrap):
        raw_deltas = []
        advantages = []
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            raw = b - a
            raw_deltas.append(raw)
            advantages.append(raw * direction)
        raw_samples.append(sum(raw_deltas) / n)
        advantage_samples.append(sum(advantages) / n)

    alpha = 1.0 - confidence
    return {
        "raw_delta_ci_low": _quantile(raw_samples, alpha / 2.0),
        "raw_delta_ci_high": _quantile(raw_samples, 1.0 - alpha / 2.0),
        "model_b_advantage_ci_low": _quantile(advantage_samples, alpha / 2.0),
        "model_b_advantage_ci_high": _quantile(advantage_samples, 1.0 - alpha / 2.0),
        "bootstrap_probability_model_b_better": (
            sum(1 for value in advantage_samples if value > practical_delta) / len(advantage_samples)
        ),
        "bootstrap_probability_model_a_better": (
            sum(1 for value in advantage_samples if value < -practical_delta) / len(advantage_samples)
        ),
    }


def _case_scores_by_metric(
    cases: list[BenchmarkCase],
    predictions_a: list[dict[str, Any] | None],
    predictions_b: list[dict[str, Any] | None],
    *,
    frame: str,
    metric_paths: tuple[str, ...],
    include_expensive_record_fields: bool,
) -> dict[str, dict[str, Any]]:
    out = {
        path: {
            "pairs": [],
            "missing_model_a": 0,
            "missing_model_b": 0,
        }
        for path in metric_paths
    }
    for case, pred_a, pred_b in zip(cases, predictions_a, predictions_b):
        score_a = score_one_case(
            case,
            pred_a,
            frame=frame,
            include_expensive_record_fields=include_expensive_record_fields,
        )
        score_b = score_one_case(
            case,
            pred_b,
            frame=frame,
            include_expensive_record_fields=include_expensive_record_fields,
        )
        for path in metric_paths:
            value_a = _metric_value(score_a, path)
            value_b = _metric_value(score_b, path)
            if value_a is None:
                out[path]["missing_model_a"] += 1
            if value_b is None:
                out[path]["missing_model_b"] += 1
            if value_a is not None and value_b is not None:
                out[path]["pairs"].append((value_a, value_b))
    return out


def _compare_scope(
    cases: list[BenchmarkCase],
    predictions_a: list[dict[str, Any] | None],
    predictions_b: list[dict[str, Any] | None],
    *,
    frame: str,
    metric_paths: tuple[str, ...],
    model_a_name: str,
    model_b_name: str,
    n_bootstrap: int,
    confidence: float,
    practical_delta: float,
    case_tie_tolerance: float,
    metric_directions: dict[str, str] | None,
    include_expensive_record_fields: bool,
    rng: random.Random,
) -> dict[str, Any]:
    per_metric = _case_scores_by_metric(
        cases,
        predictions_a,
        predictions_b,
        frame=frame,
        metric_paths=metric_paths,
        include_expensive_record_fields=include_expensive_record_fields,
    )
    rows: dict[str, dict[str, Any]] = {}
    for path in metric_paths:
        direction, direction_name = _direction_for_metric(path, metric_directions)
        info = per_metric[path]
        pairs = info["pairs"]
        a_values = [a for a, _ in pairs]
        b_values = [b for _, b in pairs]
        raw_deltas = [b - a for a, b in pairs]
        advantages = [raw * direction for raw in raw_deltas]
        advantage = _mean(advantages)

        wins_a = wins_b = ties = 0
        for value in advantages:
            if value > case_tie_tolerance:
                wins_b += 1
            elif value < -case_tie_tolerance:
                wins_a += 1
            else:
                ties += 1

        interval = _bootstrap_metric(
            pairs,
            direction=direction,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            practical_delta=practical_delta,
            rng=rng,
        )
        verdict, preferred_model = _metric_verdict(
            advantage,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            practical_delta=practical_delta,
            ci_low=interval.get("model_b_advantage_ci_low"),
            ci_high=interval.get("model_b_advantage_ci_high"),
        )
        n_compared = len(pairs)
        row = {
            "metric": path,
            "direction": direction_name,
            "model_a": _mean(a_values),
            "model_b": _mean(b_values),
            "raw_delta_model_b_minus_model_a": _mean(raw_deltas),
            "model_b_advantage": advantage,
            "practical_delta": practical_delta,
            "verdict": verdict,
            "preferred_model": preferred_model,
            "n_cases": len(cases),
            "n_compared_cases": n_compared,
            "n_missing_model_a": info["missing_model_a"],
            "n_missing_model_b": info["missing_model_b"],
            "win_loss_tie": {
                "model_a_wins": wins_a,
                "model_b_wins": wins_b,
                "ties": ties,
                "model_a_win_rate": wins_a / n_compared if n_compared else None,
                "model_b_win_rate": wins_b / n_compared if n_compared else None,
                "tie_rate": ties / n_compared if n_compared else None,
            },
        }
        row.update(interval)
        rows[path] = row
    return {
        "summary": _summarize_rows(rows),
        "metrics": rows,
    }


def build_model_comparison_report(
    cases: Iterable[BenchmarkCase],
    predictions_a: Iterable[dict[str, Any] | None],
    predictions_b: Iterable[dict[str, Any] | None],
    *,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    frame: str = "canonical",
    metric_paths: Iterable[str] | None = None,
    match_by: str | None = None,
    duplicate_policy: str = "first",
    n_bootstrap: int = 0,
    confidence: float = 0.95,
    seed: int = 123,
    include_strata: bool = True,
    min_stratum_cases: int = 2,
    practical_delta: float = 0.0,
    case_tie_tolerance: float = 0.0,
    metric_directions: dict[str, str] | None = None,
    include_expensive_record_fields: bool = True,
) -> dict[str, Any]:
    """Compare two prediction sets on the same GenAIRR benchmark cases.

    Raw deltas are reported as ``model_b - model_a``. ``model_b_advantage`` is
    direction-adjusted, so positive values always favor model B.
    """

    case_list = list(cases)
    preds_a = list(predictions_a)
    preds_b = list(predictions_b)
    match_report: dict[str, Any] = {}
    if match_by and match_by != "order":
        match_a = align_predictions_to_cases(case_list, preds_a, id_field=match_by, duplicate_policy=duplicate_policy)
        match_b = align_predictions_to_cases(case_list, preds_b, id_field=match_by, duplicate_policy=duplicate_policy)
        preds_a = match_a.predictions
        preds_b = match_b.predictions
        match_report = {
            "model_a": match_a.report,
            "model_b": match_b.report,
        }
    else:
        if len(case_list) != len(preds_a):
            raise ValueError(f"case/model A prediction length mismatch: {len(case_list)} != {len(preds_a)}")
        if len(case_list) != len(preds_b):
            raise ValueError(f"case/model B prediction length mismatch: {len(case_list)} != {len(preds_b)}")

    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    if n_bootstrap and not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if practical_delta < 0:
        raise ValueError("practical_delta must be non-negative")
    if case_tie_tolerance < 0:
        raise ValueError("case_tie_tolerance must be non-negative")

    paths = tuple(metric_paths or DEFAULT_COMPARISON_METRICS)
    rng = random.Random(seed)
    overall = _compare_scope(
        case_list,
        preds_a,
        preds_b,
        frame=frame,
        metric_paths=paths,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        practical_delta=practical_delta,
        case_tie_tolerance=case_tie_tolerance,
        metric_directions=metric_directions,
        include_expensive_record_fields=include_expensive_record_fields,
        rng=rng,
    )
    report: dict[str, Any] = {
        "comparison": {
            "method": "paired_case_delta",
            "truth_source": "GenAIRR benchmark cases",
            "model_a": model_a_name,
            "model_b": model_b_name,
            "n_cases": len(case_list),
            "frame": frame,
            "metric_paths": list(paths),
            "raw_delta": "model_b - model_a",
            "model_b_advantage": "direction-adjusted delta; positive favors model_b",
            "n_bootstrap": n_bootstrap,
            "confidence": confidence if n_bootstrap else None,
            "seed": seed if n_bootstrap else None,
            "practical_delta": practical_delta,
            "case_tie_tolerance": case_tie_tolerance,
        },
        "summary": overall["summary"],
        "overall": overall["metrics"],
        "by_stratum": {},
        "skipped_strata": [],
    }
    if match_report:
        report["prediction_matching"] = match_report

    if include_strata:
        by_stratum: dict[str, list[int]] = defaultdict(list)
        for idx, case in enumerate(case_list):
            by_stratum[case.stratum].append(idx)
        for stratum, indices in sorted(by_stratum.items()):
            if n_bootstrap and len(indices) < min_stratum_cases:
                report["skipped_strata"].append(
                    {
                        "stratum": stratum,
                        "n_cases": len(indices),
                        "reason": f"requires at least {min_stratum_cases} cases",
                    }
                )
                continue
            scope = _compare_scope(
                [case_list[i] for i in indices],
                [preds_a[i] for i in indices],
                [preds_b[i] for i in indices],
                frame=frame,
                metric_paths=paths,
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                practical_delta=practical_delta,
                case_tie_tolerance=case_tie_tolerance,
                metric_directions=metric_directions,
                include_expensive_record_fields=include_expensive_record_fields,
                rng=rng,
            )
            report["by_stratum"][stratum] = scope
    return report

