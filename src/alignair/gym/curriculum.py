"""Curriculum: training progress -> GenAIRR corruption parameters (easy -> hard)."""


def _lerp(a, b, p):
    return a + (b - a) * p


class Curriculum:
    """Ramps corruption from clean (p=0) to fully corrupted (p=1)."""

    def __init__(self, stages: int = 5):
        self.stages = stages

    def params(self, p: float) -> dict:
        p = max(0.0, min(1.0, p))
        return {
            "mutation_rate": _lerp(0.005, 0.08, p),
            "end_loss_5": (0, int(round(_lerp(0, 25, p)))),
            "end_loss_3": (0, int(round(_lerp(0, 25, p)))),
            "indel_count": (0, int(round(_lerp(0, 5, p)))),
            "seq_error_rate": _lerp(0.0, 0.02, p),
            "ambiguous_count": (0, int(round(_lerp(0, 5, p)))),
        }

    def stage(self, p: float) -> int:
        p = max(0.0, min(1.0, p))
        return min(self.stages - 1, int(p * self.stages))

    def describe(self, p: float) -> str:
        pr = self.params(p)
        return (f"curriculum stage {self.stage(p) + 1}/{self.stages} (p={p:.2f}): "
                f"mut≤{pr['mutation_rate']:.3f}, trim≤{pr['end_loss_5'][1]}, "
                f"indel≤{pr['indel_count'][1]}, seq_err≤{pr['seq_error_rate']:.3f}, "
                f"N≤{pr['ambiguous_count'][1]}")
