"""Base curriculum implementation containing the basic scalar ramp curriculum."""
import math
from numbers import Real

from ....errors import ConfigurationError

# Polymerase-slippage indels are rare in real reads (GenAIRR: under 1% per read at standard fidelity), so the
# curriculum gives them only the share of reads the indel head needs to learn from: the share of reads with at
# least one indel ramps linearly from INDEL_FRACTION[0] at p=0 to INDEL_FRACTION[1] at p=1. Over the default
# training mix (``streams.CurriculumStreamPolicy``) about 85% of reads carry no indel (84.4% of 7,000 records).
INDEL_FRACTION = (0.05, 0.25)
# Among reads with indels: one indel dominates, then two, rarely three.
INDEL_COUNT_WEIGHTS = ((1, 0.75), (2, 0.20), (3, 0.05))


def _lerp(a, b, p):
    return a + (b - a) * p


def indel_fraction_range(value) -> tuple:
    """Validate the ``(low, high)`` share of reads with an indel: finite, ``0 <= low <= high <= 1``."""
    try:
        if isinstance(value, (str, bytes)):
            raise TypeError
        low, high = value
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("indel_fraction must be a (low, high) pair", context={"field": "indel_fraction"}) from exc
    if (any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in (low, high))
            or not 0 <= low <= high <= 1):
        raise ConfigurationError("indel_fraction must satisfy 0 <= low <= high <= 1",
                                 context={"field": "indel_fraction"})
    return float(low), float(high)


class Curriculum:
    """Ramps corruption from clean (p=0) to fully corrupted (p=1).

    ``indel_fraction`` is the ``(low, high)`` share of reads with at least one polymerase indel at
    p=0 and p=1 (default ``INDEL_FRACTION``)."""

    def __init__(self, stages: int = 5, indel_fraction=INDEL_FRACTION):
        self.stages = stages
        self.indel_fraction = indel_fraction_range(indel_fraction)

    def indel_share(self, p: float) -> float:
        """The share of reads with at least one indel at progress ``p``."""
        return _lerp(*self.indel_fraction, max(0.0, min(1.0, p)))

    def indel_count(self, p: float) -> tuple:
        """GenAIRR's weighted ``(count, weight)`` form: no indel for ``1 - indel_share(p)`` of reads, else 1, 2 or
        3 indels by ``INDEL_COUNT_WEIGHTS``. Counts of zero weight are left out."""
        share = self.indel_share(p)
        pairs = ((0, 1.0 - share),) + tuple((count, share * weight) for count, weight in INDEL_COUNT_WEIGHTS)
        return tuple(pair for pair in pairs if pair[1] > 0)

    def params(self, p: float) -> dict:
        p = max(0.0, min(1.0, p))
        return {
            "mutation_rate": _lerp(0.005, 0.15, p),
            # Minor observation-stage end loss (primer/read-end nibble) — the CORRUPTION axis, kept
            # small so the bulk of the molecule survives. Read LENGTH / short-read amplicons are a
            # separate, orthogonal axis modelled by GenAIRR end-loss profiles in the training mix
            # (V-anchored / J-anchored / fragment); see train.gym.streams.CurriculumStreamPolicy.
            "end_loss_5": (0, int(round(_lerp(0, 40, p)))),
            "end_loss_3": (0, int(round(_lerp(0, 40, p)))),
            "indel_count": self.indel_count(p),
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
                f"indel {100 * self.indel_share(p):.0f}%, seq_err≤{pr['seq_error_rate']:.3f}, "
                f"N≤{pr['ambiguous_count'][1]}, orient({100*pr['orient_prob']:.0f}%)")
