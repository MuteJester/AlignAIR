"""PromotionGate: all per-head locks must open (at the level's threshold) to climb."""
from typing import Sequence

from .config import GateSpec
from .state import GateStatus


class PromotionGate:
    def __init__(self, gates: Sequence[GateSpec]):
        self.gates = tuple(gates)

    def statuses(self, metrics: dict, level: int) -> list:
        out = []
        for spec in self.gates:
            thr = spec.thresholds[max(0, min(len(spec.thresholds) - 1, level))]
            if spec.metric in metrics:
                val = float(metrics[spec.metric])
            else:
                # unmeasured => force CLOSED so we never promote on missing evidence
                val = 0.0 if spec.direction == "higher" else float("inf")
            out.append(GateStatus(spec.metric, val, thr, spec.direction))
        return out

    def evaluate(self, metrics: dict, level: int) -> tuple:
        sts = self.statuses(metrics, level)
        blocking = [s.name for s in sts if not s.is_open]
        return (len(blocking) == 0, blocking)

    def should_demote(self, metrics: dict, level: int, margin: float) -> tuple[bool, list]:
        """Check if any metric has dropped below the demotion threshold.
        Returns (should_demote: bool, failing_gates: list[str])."""
        failing = []
        for spec in self.gates:
            thr = spec.thresholds[max(0, min(len(spec.thresholds) - 1, level))]
            if spec.metric in metrics:
                val = float(metrics[spec.metric])
            else:
                # Ignore missing metrics for demotion
                continue
            if spec.direction == "higher":
                if val < thr - margin:
                    failing.append(spec.metric)
            else:  # "lower"
                if val > thr + margin:
                    failing.append(spec.metric)
        return len(failing) > 0, failing
