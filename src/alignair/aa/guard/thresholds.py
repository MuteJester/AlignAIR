"""Stage-6 Task 6.1 (implemented early per the corrected Stage-5 sequence) — exact Clopper-Pearson bounds +
guardband threshold-grid selection.

The pre-unlock winner's-curse guardband requires the ONE-SIDED exact upper bound (not the point estimate) to
sit at <=80% of the release limit: FAR upper <= 4% for a 5% gate / <= 8% for a 10% gate, and FRR upper <= 8%.
A candidate's operating point is chosen from a FROZEN finite grid on the fresh selection subset only.
"""
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class Thresholds:
    candidate: str
    grid_point: dict            # the frozen operating point (e.g. score cutoffs)
    far_upper: float
    frr_upper: float
    n_neg: int
    n_pos: int
    gate_kind: str              # "gate_5pct" | "gate_10pct"


def _far_limit(gate_kind: str) -> float:
    return FAR_LIMIT_5PCT if gate_kind == "gate_5pct" else FAR_LIMIT_10PCT


def passes_guardband(far_upper: float, frr_upper: float, gate_kind: str) -> bool:
    """FAR upper <= gate ceiling AND FRR upper <= 8%."""
    return far_upper <= _far_limit(gate_kind) + 1e-12 and frr_upper <= FRR_LIMIT + 1e-12


def select_thresholds(candidate, selection_subset, grid, *, gate_kind="gate_5pct") -> Thresholds | None:
    """Choose the frozen operating point from `grid` on the fresh SELECTION subset only.

    `grid`: iterable of (grid_point, far_k, far_n, frr_k, frr_n). Returns the first grid point (in the grid's
    predeclared order) whose exact upper bounds pass the guardband, else None (no admissible point)."""
    for grid_point, far_k, far_n, frr_k, frr_n in grid:
        fu = clopper_pearson_upper(far_k, far_n)
        fr = clopper_pearson_upper(frr_k, frr_n)
        if passes_guardband(fu, fr, gate_kind):
            return Thresholds(candidate=candidate, grid_point=dict(grid_point), far_upper=round(fu, 5),
                              frr_upper=round(fr, 5), n_neg=far_n, n_pos=frr_n, gate_kind=gate_kind)
    return None
