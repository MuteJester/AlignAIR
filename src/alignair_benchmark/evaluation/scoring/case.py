from __future__ import annotations

from typing import Any

from ...core.schema import BenchmarkCase
from .genes import score_gene
from .registry import GLOBAL_SCORING_COMPONENTS


def score_one_case(
    case: BenchmarkCase,
    prediction: dict[str, Any] | None,
    *,
    frame: str = "canonical",
    include_expensive_record_fields: bool = True,
) -> dict[str, Any]:
    """Score one prediction against one benchmark case."""

    pred = prediction or {}
    global_metrics: dict[str, float] = {}
    gene_metrics: dict[str, dict[str, float]] = {}
    for component in GLOBAL_SCORING_COMPONENTS:
        global_metrics.update(component.score(pred, case, frame))
    alt_frame = "presented" if frame == "canonical" else "canonical"
    for gene, truth in case.truth(frame).items():
        gene_metrics[gene] = score_gene(
            pred,
            truth,
            gene,
            case=case,
            alt_truth=case.truth(alt_frame).get(gene),
            include_expensive_record_fields=include_expensive_record_fields,
        )
    return {"global": global_metrics, "genes": gene_metrics}
