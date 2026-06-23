"""FactoredCurriculum: a per-axis competence-paced generalization of
StratifiedCurriculum. Each difficulty axis advances on its OWN clock (pace), so the
axes decouple — no shortcut learning, and the full-length-heavy-SHM corner becomes
reachable. Reuses StratifiedCurriculum's GenAIRR-faithful bin profiles; terminal
(all paces=1) equals the deployment distribution incl. the hard tail."""
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
