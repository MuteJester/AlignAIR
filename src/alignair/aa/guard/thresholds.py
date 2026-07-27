"""Stage-6 Task 6.1 (implemented early per the corrected Stage-5 sequence) — exact Clopper-Pearson bound
PRIMITIVES + guardband ceilings.

The pre-unlock winner's-curse guardband requires the ONE-SIDED exact upper bound (not the point estimate) to
sit at <=80% of the release limit: FAR upper <= 4% for a 5% gate / <= 8% for a 10% gate, and FRR upper <= 8%.

Threshold SELECTION (deriving per-group failure counts internally from scored records, unit aggregation, and
the multi-group gate) lives in ``selection.py``; this module only holds the exact-bound math + ceilings so both
the selector and the tests share one source of truth.
"""
from __future__ import annotations

from scipy.stats import beta

FAR_LIMIT_5PCT = 0.04         # winner's-curse guardband upper-bound ceilings
FAR_LIMIT_10PCT = 0.08
FRR_LIMIT = 0.08


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided exact (Clopper-Pearson) upper bound on a binomial rate given k failures in n trials."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def far_limit(gate_kind: str) -> float:
    return FAR_LIMIT_5PCT if gate_kind == "gate_5pct" else FAR_LIMIT_10PCT


def ceiling_for(kind: str, gate_kind: str) -> float:
    """The guardband ceiling for a group's exact upper bound: FAR uses the gate ceiling, FRR uses 8%."""
    return far_limit(gate_kind) if kind == "far" else FRR_LIMIT


def passes_guardband(far_upper: float, frr_upper: float, gate_kind: str) -> bool:
    """FAR upper <= gate ceiling AND FRR upper <= 8%."""
    return far_upper <= far_limit(gate_kind) + 1e-12 and frr_upper <= FRR_LIMIT + 1e-12
