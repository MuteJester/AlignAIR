"""FactoredCurriculum implementation."""

from .base import _lerp
from .stratified import StratifiedCurriculum

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


# the axis-isolated FrozenLattice cell that stresses each axis
_AXIS_CELL = {
    "mutation_count": "heavy_shm_fulllen",
    "end_loss_5": "trim", "end_loss_3": "trim",
    "indel_count": "indel", "ambiguous_count": "ambiguous",
    "seq_error_rate": "seq_error", "crop": "fragment", "orient": "orient",
}


def axis_competence_from_field(field: dict, fallback_cell: str = "clean",
                               use_lcb: bool = False) -> dict:
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
