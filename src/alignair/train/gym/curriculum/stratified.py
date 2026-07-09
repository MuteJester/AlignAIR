"""StratifiedCurriculum implementation."""

from .base import _lerp


class StratifiedCurriculum:
    """Decoupled, full-range difficulty MIXTURE (replaces the single-scalar ramp)."""

    # per axis: bins + (easy-weighted, hard-weighted) probabilities.
    _MIX = {
        "mutation_count": ([0, 3, 8, 18, 35, 55, 80],
                           [0.55, 0.25, 0.12, 0.05, 0.02, 0.01, 0.00],
                           [0.20, 0.12, 0.15, 0.18, 0.18, 0.12, 0.05]),
        "end_loss_5":     ([0, 10, 30, 70, 120],
                           [0.85, 0.10, 0.04, 0.01, 0.00],
                           [0.55, 0.15, 0.15, 0.10, 0.05]),
        "end_loss_3":     ([0, 10, 25, 45],
                           [0.88, 0.09, 0.02, 0.01],
                           [0.65, 0.20, 0.10, 0.05]),
        "indel_count":    ([0, 2, 5, 9],
                           [0.95, 0.04, 0.01, 0.00],
                           [0.78, 0.12, 0.07, 0.03]),
        "ambiguous_count": ([0, 3, 10],
                            [0.95, 0.04, 0.01],
                            [0.80, 0.15, 0.05]),
    }

    @staticmethod
    def _blend(bins, easy, hard, tau, eps=1e-3):
        # floor each weight > 0 (GenAIRR's EmpiricalLengthDist rejects zero weights)
        w = [max((1 - tau) * e + tau * h, eps) for e, h in zip(easy, hard)]
        s = sum(w)
        return [(b, wi / s) for b, wi in zip(bins, w)]

    def params(self, p: float = 1.0) -> dict:
        tau = max(0.0, min(1.0, p))
        out = {k: self._blend(*v, tau) for k, v in self._MIX.items()}
        out.update({
            "seq_error_rate": _lerp(0.001, 0.01, tau),
            "crop_prob": _lerp(0.1, 0.5, tau),
            "crop_len_min": 50, "crop_len_max": 576, "crop_log_uniform": True,
            "orient_prob": _lerp(0.1, 0.35, tau),
        })
        return out

    def stage(self, p: float) -> int:
        return min(4, int(max(0.0, min(1.0, p)) * 5))

    def describe(self, p: float) -> str:
        return (f"stratified mixture (tau={max(0.0,min(1.0,p)):.2f}; decoupled axes, "
                f"easy-first->full-range SHM, naive mass always present)")
