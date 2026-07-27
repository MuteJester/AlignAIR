"""Stage-5 Task 5.3 — multi-group threshold selection with counts derived INTERNALLY.

Threshold selection may never trust caller-supplied counts. For every proposed grid point it:
  1. applies the candidate decision graph (``apply_decision``) to each BLIND scored record,
  2. maps each record to its selection group and predeclared independence unit,
  3. aggregates to unit-level outcomes (conservative: a unit fails if ANY member fails — no pooling across
     groups, no compensation), and
  4. requires EVERY applicable group gate to pass independently (exact Clopper-Pearson upper bound <= ceiling).

The seven gates: IGK/IGL, TCR, VHH/non-human, IgSF (FAR); Structural aggregated by parent lineage (FAR);
Real-positive and Simulator-positive (FRR). A grid point is admissible only if all required groups pass; if
none does, selection returns ``None`` (a STOP).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from alignair.aa.guard.decision import CandidateSpec, apply_decision, candidate_grid
from alignair.aa.guard.thresholds import ceiling_for, clopper_pearson_upper

# group -> (error kind, guardband gate, predeclared independence-unit field). Keyed by cohort group.
SELECTION_GATES = {
    "igk_igl":       {"kind": "far", "gate_kind": "gate_5pct",  "unit": "lineage"},
    "tcr":           {"kind": "far", "gate_kind": "gate_5pct",  "unit": "lineage"},
    "vhh_nonhuman":  {"kind": "far", "gate_kind": "gate_10pct", "unit": "lineage"},
    "igsf":          {"kind": "far", "gate_kind": "gate_10pct", "unit": "lineage"},
    "structural":    {"kind": "far", "gate_kind": "gate_5pct",  "unit": "parent_lineage"},   # by parent lineage
    "positive_real": {"kind": "frr", "gate_kind": "frr",        "unit": "lineage"},
    "positive_sim":  {"kind": "frr", "gate_kind": "frr",        "unit": "lineage"},
}
# The unit-reduction is conservative: a unit fails if ANY of its records fails (upper-bounds every rate).
UNIT_REDUCTION = "any_member_fails"


@dataclass(frozen=True)
class GroupResult:
    group: str
    kind: str
    gate_kind: str
    unit: str
    k: int                  # failing units
    n: int                  # total units (after independence aggregation)
    estimate: float
    upper: float            # exact one-sided upper bound
    ceiling: float
    passed: bool


@dataclass(frozen=True)
class ThresholdSelection:
    candidate: str
    grid_point: dict
    per_group: dict         # group -> GroupResult (as dict)
    gates_pass: bool


def _unit_id(rs, unit_field: str) -> str:
    u = rs.unit or {}
    return str(u.get(unit_field) or u.get("lineage") or rs.record_id_sha256)


def group_counts(blind_scores, spec: CandidateSpec, grid_point: dict) -> dict:
    """Per-group unit-aggregated failure counts + exact bounds for one grid point (derived internally)."""
    units = defaultdict(lambda: defaultdict(list))                   # group -> unit_id -> [record_failed?]
    for rs in blind_scores:
        gate = SELECTION_GATES.get(rs.group)
        if gate is None:
            continue                                                # not a selection-gated group
        dec = apply_decision(rs, spec, grid_point)
        failed = dec.accepted if gate["kind"] == "far" else (not dec.accepted)
        units[rs.group][_unit_id(rs, gate["unit"])].append(bool(failed))
    table = {}
    for g, unit_map in units.items():
        gate = SELECTION_GATES[g]
        unit_failed = [any(members) for members in unit_map.values()]   # conservative unit reduction
        n = len(unit_failed)
        k = sum(unit_failed)
        upper = clopper_pearson_upper(k, n)
        ceiling = ceiling_for(gate["kind"], gate["gate_kind"])
        table[g] = GroupResult(group=g, kind=gate["kind"], gate_kind=gate["gate_kind"], unit=gate["unit"],
                               k=k, n=n, estimate=round(k / n, 6) if n else 0.0, upper=round(upper, 6),
                               ceiling=ceiling, passed=bool(upper <= ceiling + 1e-12))
    return table


def passes_all_gates(table: dict, required_groups) -> bool:
    """Every required group must be present AND pass its own gate — no pooling, no compensation."""
    for g in required_groups:
        gr = table.get(g)
        if gr is None or not gr.passed:
            return False
    return True


def select_thresholds(spec: CandidateSpec, blind_scores, required_groups, grid=None) -> "ThresholdSelection | None":
    """First admissible grid point (candidate grid, permissive-first) whose EVERY required group gate passes.
    Counts are derived internally from the blind scores; returns None (STOP) if no point is admissible."""
    grid = grid if grid is not None else candidate_grid(spec.cid)
    for gp in grid:
        table = group_counts(blind_scores, spec, gp)
        if passes_all_gates(table, required_groups):
            return ThresholdSelection(candidate=spec.cid, grid_point=dict(gp),
                                      per_group={g: _gr_dict(table[g]) for g in table}, gates_pass=True)
    return None


def _gr_dict(gr: GroupResult) -> dict:
    return {"group": gr.group, "kind": gr.kind, "gate_kind": gr.gate_kind, "unit": gr.unit, "k": gr.k,
            "n": gr.n, "estimate": gr.estimate, "upper": gr.upper, "ceiling": gr.ceiling, "passed": gr.passed}
