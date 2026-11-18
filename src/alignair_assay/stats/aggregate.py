"""Coverage-aware aggregation of per-case scores.

Per-case convention: an absent key means the metric does not apply to the case; ``nan`` means it applies but the
tool did not cover it; any finite value counts toward the mean.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from alignair_assay.metrics import CaseScores


@dataclass(frozen=True)
class MetricSummary:
    mean: float | None
    n: int
    n_applicable: int
    coverage: float


def summarize(scores: dict[str, CaseScores], case_ids: list[str],
              applicable: dict[str, set[str]] | None = None) -> dict[str, MetricSummary]:
    """Mean, value count, applicable count and coverage per metric over ``case_ids``.

    Case ids missing from ``scores`` contribute nothing. ``applicable`` marks extra cases as applicable (and
    uncovered unless they carry a finite value) for the listed metrics.
    """
    ids = [c for c in case_ids if c in scores]
    values: dict[str, list[float]] = {}
    app: dict[str, set[str]] = {}
    for cid in ids:
        for name, v in scores[cid].items():
            app.setdefault(name, set()).add(cid)
            if math.isfinite(v):
                values.setdefault(name, []).append(v)
    if applicable:
        wanted = set(case_ids)
        for name, cids in applicable.items():
            app.setdefault(name, set()).update(c for c in cids if c in wanted)
    out: dict[str, MetricSummary] = {}
    for name in sorted(app):
        vals = values.get(name, [])
        n_app = len(app[name])
        out[name] = MetricSummary(
            mean=math.fsum(vals) / len(vals) if vals else None,
            n=len(vals), n_applicable=n_app,
            coverage=len(vals) / n_app if n_app else 0.0)
    return out


def by_slice(scores: dict[str, CaseScores], slice_of: dict[str, str]) -> dict[str, dict[str, MetricSummary]]:
    """``summarize`` per slice label; case ids with no label are skipped."""
    groups: dict[str, list[str]] = {}
    for cid in scores:
        label = slice_of.get(cid)
        if label is not None:
            groups.setdefault(label, []).append(cid)
    return {label: summarize(scores, cids) for label, cids in sorted(groups.items())}
