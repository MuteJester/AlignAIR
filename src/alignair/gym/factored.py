"""FactoredCurriculum: a per-axis competence-paced generalization of
StratifiedCurriculum. Each difficulty axis advances on its OWN clock (pace), so the
axes decouple — no shortcut learning, and the full-length-heavy-SHM corner becomes
reachable. Reuses StratifiedCurriculum's GenAIRR-faithful bin profiles.

KNOWN DIVERGENCE (tracked): the terminal (all paces=1) equals StratifiedCurriculum(1.0)
— seq_error 0.01, orient 0.35, SHM as a count mixture — which is NOT identical to the
instrument's TaskSpace.deployment() endpoint (seq_error 0.02, orient 0.5, mutation_rate
0.30). The two are different encodings of "deployment"; the instrument grades a strict
SUPERSET of the training endpoint, so passing the gate is CONSERVATIVE. Reconciling the
two encodings (count-mixture vs rate) is deferred to a follow-up; until then, treat the
instrument as the harder bar."""
from .curriculum import StratifiedCurriculum, _lerp

_MIX_AXES = ("mutation_count", "end_loss_5", "end_loss_3", "indel_count", "ambiguous_count")
_SCALAR_AXES = ("seq_error_rate", "crop", "orient")


class FactoredCurriculum:
    def __init__(self, start_pace: float = 0.1):
        self.axes = _MIX_AXES + _SCALAR_AXES
        self.pace = {a: float(start_pace) for a in self.axes}

    def _p(self, axis: str) -> float:
        return max(0.0, min(1.0, self.pace[axis]))

    def params(self, p=None) -> dict:
        mix = StratifiedCurriculum._MIX
        out = {k: StratifiedCurriculum._blend(*mix[k], self._p(k)) for k in _MIX_AXES}
        out.update({
            "seq_error_rate": _lerp(0.001, 0.01, self._p("seq_error_rate")),
            "crop_prob": _lerp(0.1, 0.5, self._p("crop")),
            "crop_len_min": 50, "crop_len_max": 576, "crop_log_uniform": True,
            "orient_prob": _lerp(0.1, 0.35, self._p("orient")),
        })
        return out

    def is_terminal(self) -> bool:
        return all(self.pace[a] >= 1.0 for a in self.axes)

    def stage(self, p=None) -> int:
        return min(4, int(max(self._p(a) for a in self.axes) * 5))

    def describe(self, p=None) -> str:
        worst = min(self.pace.values())
        hot = min(self.pace, key=self.pace.get)
        return (f"factored curriculum (per-axis pace; min={worst:.2f} on '{hot}'; "
                f"terminal={self.is_terminal()})")

    def advance(self, axis_competence: dict, threshold: float = 0.7,
                step: float = 0.1) -> list:
        moved = []
        for axis in self.axes:
            c = axis_competence.get(axis)
            if c is not None and c >= threshold and self.pace[axis] < 1.0:
                self.pace[axis] = min(1.0, self.pace[axis] + step)
                moved.append(axis)
        return moved


# the axis-isolated FrozenLattice cell that stresses each axis (one clean per-axis
# competence signal per axis). Missing cells fall back to "clean" (overall competence).
_AXIS_CELL = {
    "mutation_count": "heavy_shm_fulllen",   # SHM, full-length (the contested corner)
    "end_loss_5": "trim", "end_loss_3": "trim",
    "indel_count": "indel", "ambiguous_count": "ambiguous",
    "seq_error_rate": "seq_error", "crop": "fragment", "orient": "orient",
}


def axis_competence_from_field(field: dict, fallback_cell: str = "clean",
                               use_lcb: bool = False) -> dict:
    # use_lcb: advance a pace only when the CONSERVATIVE competence (bootstrap-CI lower
    # bound `lo`) clears the bar, so a single lucky exam can't promote (Phase-5 rigor).
    key = "lo" if use_lcb else "S"

    def _S(cell):
        if cell in field:
            return float(field[cell].get(key, field[cell].get("S", 0.0)))
        if fallback_cell in field:
            return float(field[fallback_cell].get(key, field[fallback_cell].get("S", 0.0)))
        return 0.0
    fb = _S(fallback_cell)
    out = {}
    for axis in (*_MIX_AXES, *_SCALAR_AXES):
        out[axis] = _S(_AXIS_CELL[axis]) if axis in _AXIS_CELL else fb
    return out
