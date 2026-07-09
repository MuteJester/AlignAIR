"""Base curriculum implementation containing the basic scalar ramp curriculum."""


def _lerp(a, b, p):
    return a + (b - a) * p


class Curriculum:
    """Ramps corruption from clean (p=0) to fully corrupted (p=1)."""

    def __init__(self, stages: int = 5):
        self.stages = stages

    def params(self, p: float) -> dict:
        p = max(0.0, min(1.0, p))
        return {
            "mutation_rate": _lerp(0.005, 0.15, p),
            # Minor observation-stage end loss (primer/read-end nibble) — the CORRUPTION axis, kept
            # small so the bulk of the molecule survives. Read LENGTH / short-read amplicons are a
            # separate, orthogonal axis modelled by GenAIRR end-loss profiles in the training mix
            # (V-anchored / J-anchored / fragment); see alignair_trainer._amplicon_specs.
            "end_loss_5": (0, int(round(_lerp(0, 40, p)))),
            "end_loss_3": (0, int(round(_lerp(0, 40, p)))),
            "indel_count": (0, int(round(_lerp(0, 5, p)))),
            "seq_error_rate": _lerp(0.0, 0.02, p),
            "ambiguous_count": (0, int(round(_lerp(0, 5, p)))),
            "crop_prob": 0.0,   # post-hoc gym crop retired; read length is GenAIRR end-loss (above)
            # fraction of reads presented in a non-forward orientation (revcomp /
            # complement / reverse); the model must detect and canonicalize them.
            "orient_prob": _lerp(0.0, 0.5, p),
        }

    def stage(self, p: float) -> int:
        p = max(0.0, min(1.0, p))
        return min(self.stages - 1, int(p * self.stages))

    def describe(self, p: float) -> str:
        pr = self.params(p)
        return (f"curriculum stage {self.stage(p) + 1}/{self.stages} (p={p:.2f}): "
                f"mut≤{pr['mutation_rate']:.3f}, end_loss≤{pr['end_loss_5'][1]}/{pr['end_loss_3'][1]}, "
                f"indel≤{pr['indel_count'][1]}, seq_err≤{pr['seq_error_rate']:.3f}, "
                f"N≤{pr['ambiguous_count'][1]}, orient({100*pr['orient_prob']:.0f}%)")
