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
            "mutation_rate": _lerp(0.005, 0.15, p),
            "end_loss_5": (0, int(round(_lerp(0, 25, p)))),
            "end_loss_3": (0, int(round(_lerp(0, 25, p)))),
            "indel_count": (0, int(round(_lerp(0, 5, p)))),
            "seq_error_rate": _lerp(0.0, 0.02, p),
            "ambiguous_count": (0, int(round(_lerp(0, 5, p)))),
            # fragment cropping: at high p a growing fraction of reads are cropped to
            # a junction-centered window as short as ~50bp (CDR3+flanks). The full
            # read is always retained for the rest, so every batch spans the spectrum.
            "crop_prob": _lerp(0.0, 0.6, p),
            "crop_len_min": 50,
            "crop_len_max": int(round(_lerp(576, 80, p))),
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
                f"mut≤{pr['mutation_rate']:.3f}, trim≤{pr['end_loss_5'][1]}, "
                f"indel≤{pr['indel_count'][1]}, seq_err≤{pr['seq_error_rate']:.3f}, "
                f"N≤{pr['ambiguous_count'][1]}, "
                f"crop({100*pr['crop_prob']:.0f}%≥{pr['crop_len_min']}..{pr['crop_len_max']}bp)")


class StratifiedCurriculum:
    """Decoupled, full-range difficulty MIXTURE (replaces the single-scalar ramp).

    Every axis is sampled per-read across its full range *every batch* via GenAIRR's
    per-read distribution form (mutate(count=[(n,p)...]), trims/indels/N as (value,prob)
    lists), with crop/orientation drawn per-read in the gym. So a single batch spans
    naive full reads -> hypermutated ~50bp fragments, with the axes DECOUPLED (a
    low-SHM fragment and a high-SHM full read both occur) — the regimes the scalar-p
    ramp could never produce. SHM uses mutation COUNT (rate is scalar-only in GenAIRR);
    the count distribution is bimodal-ish: a naive spike + a broad hypermutated tail.
    Difficulty does not ramp with p (full spectrum from step 0); easy mass (naive) is
    always present so nothing is forgotten."""

    def params(self, p: float = 1.0) -> dict:
        return {
            # per-read SHM via mutation-count categorical: naive spike + heavy tail
            "mutation_count": [(0, 0.20), (3, 0.12), (8, 0.15), (18, 0.18),
                               (35, 0.18), (55, 0.12), (80, 0.05)],
            "end_loss_5": [(0, 0.6), (20, 0.15), (60, 0.15), (120, 0.10)],
            "end_loss_3": [(0, 0.7), (15, 0.2), (40, 0.10)],
            "indel_count": [(0, 0.8), (2, 0.1), (5, 0.07), (9, 0.03)],
            "seq_error_rate": 0.005,
            "ambiguous_count": [(0, 0.8), (3, 0.15), (10, 0.05)],
            "crop_prob": 0.5, "crop_len_min": 50, "crop_len_max": 576, "crop_log_uniform": True,
            "orient_prob": 0.3,
        }

    def stage(self, p: float) -> int:
        return 0

    def describe(self, p: float) -> str:
        return "stratified full-range mixture (decoupled axes; SHM count-distribution naive->hypermutated)"
