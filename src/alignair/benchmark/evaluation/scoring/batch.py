from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ...core.schema import BenchmarkCase, GENES
from .primitives import avg
from .case import score_one_case


def _score_cases(
    cases: Iterable[BenchmarkCase],
    predictions: Iterable[dict[str, Any]],
    *,
    frame: str = "canonical",
    include_strata: bool = True,
    include_expensive_record_fields: bool = True,
) -> dict[str, Any]:
    """Score predictions against benchmark cases.

    ``frame`` controls whether coordinates/labels are compared to canonical or
    presented-frame truth.
    """

    cases = list(cases)
    predictions = list(predictions)
    if len(cases) != len(predictions):
        raise ValueError(f"case/prediction length mismatch: {len(cases)} != {len(predictions)}")

    gene_metrics: dict[str, dict[str, list[float]]] = {g: defaultdict(list) for g in GENES}
    global_metrics: dict[str, list[float]] = defaultdict(list)
    by_stratum_cases: dict[str, list[BenchmarkCase]] = defaultdict(list)
    by_stratum_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case, pred in zip(cases, predictions):
        by_stratum_cases[case.stratum].append(case)
        by_stratum_preds[case.stratum].append(pred or {})
        one = score_one_case(
            case,
            pred,
            frame=frame,
            include_expensive_record_fields=include_expensive_record_fields,
        )
        for k, v in one["global"].items():
            global_metrics[k].append(v)
        for gene, vals in one["genes"].items():
            for k, v in vals.items():
                gene_metrics[gene][k].append(v)

    out = {
        "n_cases": len(cases),
        "frame": frame,
        "global": {k: avg(v) for k, v in sorted(global_metrics.items())},
        "genes": {
            gene: {k: avg(v) for k, v in sorted(vals.items())}
            for gene, vals in gene_metrics.items()
        },
    }
    if include_strata:
        out["by_stratum"] = {
            name: _score_cases(
                by_stratum_cases[name],
                by_stratum_preds[name],
                frame=frame,
                include_strata=False,
                include_expensive_record_fields=include_expensive_record_fields,
            )
            for name in sorted(by_stratum_cases)
        }
    else:
        out["by_stratum"] = {}
    return out


def score_cases(
    cases: Iterable[BenchmarkCase],
    predictions: Iterable[dict[str, Any]],
    *,
    frame: str = "canonical",
    include_strata: bool = True,
    include_expensive_record_fields: bool = True,
) -> dict[str, Any]:
    """Score predictions against benchmark cases."""

    return _score_cases(
        cases,
        predictions,
        frame=frame,
        include_strata=include_strata,
        include_expensive_record_fields=include_expensive_record_fields,
    )
