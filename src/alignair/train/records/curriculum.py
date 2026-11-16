"""Base curriculum implementation containing the basic scalar ramp curriculum."""
import math
from numbers import Real

from ...errors import ConfigurationError

# How a curriculum draws its polymerase-indel count per read:
# - "uniform", the default: a uniform count in [0, round(5p)]. The benchmark and evaluation
#   strata are generated from it, so it never changes.
# - "weighted": indels are rare in real reads (GenAIRR: under 1% per read at standard fidelity), so this form gives
#   them only the share of reads the indel head needs to learn from. The weight of reads with at least one indel
#   ramps linearly from INDEL_FRACTION[0] at p=0 to INDEL_FRACTION[1] at p=1, and INDEL_COUNT_WEIGHTS splits it
#   over 1, 2 and 3 indels. AlignAIR training (``streams.CurriculumStreamPolicy``) uses it: over the default
#   training mix about 85% of reads carry no indel (84.4% of the first 7,000 records at seed 0).
INDEL_FORMS = ("uniform", "weighted")
INDEL_FRACTION = (0.05, 0.25)
# Requested weights of 1, 2 and 3 indels among the weight of reads with indels; GenAIRR draws the realised counts.
INDEL_COUNT_WEIGHTS = ((1, 0.75), (2, 0.20), (3, 0.05))
# GenAIRR draws a sequencing-error, PCR-error or uniform-mutate replacement base from the 3 other bases, so the realised
# rate is the requested one. The chip's sequencing-error level is about 1/1.32 of the requested rate (the level of a
# draw from all 4 bases, some of them silent), so the curriculum scales its sequencing-error rate by this; S5F mutation
# is unchanged, and the curriculum sets no PCR or uniform-mutate rate.
SEQUENCING_ERROR_SCALE = 0.757


def _lerp(a, b, p):
    return a + (b - a) * p


def indel_form_name(value) -> str:
    """Validate a curriculum indel form: one of ``INDEL_FORMS``."""
    if not isinstance(value, str) or value not in INDEL_FORMS:
        raise ConfigurationError(f"indel_form must be one of {INDEL_FORMS}", context={"field": "indel_form"})
    return value


def indel_fraction_range(value) -> tuple:
    """Validate the ``(low, high)`` weight of reads with an indel: finite, ``0 <= low <= high <= 1``."""
    try:
        if isinstance(value, (str, bytes)):
            raise TypeError
        low, high = value
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("indel_fraction must be a (low, high) pair",
                                 context={"field": "indel_fraction"}) from exc
    if (any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in (low, high))
            or not 0 <= low <= high <= 1):
        raise ConfigurationError("indel_fraction must satisfy 0 <= low <= high <= 1",
                                 context={"field": "indel_fraction"})
    return float(low), float(high)


class Curriculum:
    """Ramps corruption from clean (p=0) to fully corrupted (p=1).

    ``indel_form`` (``INDEL_FORMS``) selects how the polymerase-indel count is drawn: the ``"uniform"`` range (the
    default) or the ``"weighted"`` form, whose weight of reads with at least one
    indel is ``indel_fraction`` ``(low, high)`` at p=0 and p=1 (default ``INDEL_FRACTION``)."""

    def __init__(self, stages: int = 5, indel_form: str = "uniform", indel_fraction=INDEL_FRACTION):
        self.stages = stages
        self.indel_form = indel_form_name(indel_form)
        self.indel_fraction = indel_fraction_range(indel_fraction)

    def indel_share(self, p: float) -> float:
        """The weighted form's weight of reads with at least one indel at progress ``p``."""
        return _lerp(*self.indel_fraction, max(0.0, min(1.0, p)))

    def indel_count(self, p: float) -> tuple:
        """GenAIRR's ``count`` for progress ``p``. Uniform: the ``(0, round(5p))`` range. Weighted: ``(count,
        weight)`` pairs, no indel with weight ``1 - indel_share(p)``, else 1, 2 or 3 indels by
        ``INDEL_COUNT_WEIGHTS``; counts of zero weight are left out."""
        p = max(0.0, min(1.0, p))
        if self.indel_form == "uniform":
            return (0, round(_lerp(0, 5, p)))
        share = self.indel_share(p)
        pairs = ((0, 1.0 - share),) + tuple((count, share * weight) for count, weight in INDEL_COUNT_WEIGHTS)
        return tuple(pair for pair in pairs if pair[1] > 0)

    def params(self, p: float) -> dict:
        """The GenAIRR corruption settings at progress ``p`` (clamped to [0, 1])."""
        p = max(0.0, min(1.0, p))
        return {
            "mutation_rate": _lerp(0.005, 0.15, p),
            # Minor observation-stage end loss (primer/read-end nibble) — the CORRUPTION axis, kept
            # small so the bulk of the molecule survives. Read LENGTH / short-read amplicons are a
            # separate, orthogonal axis modelled by GenAIRR end-loss profiles in the training mix
            # (V-anchored / J-anchored / fragment); see train.records.streams.CurriculumStreamPolicy.
            "end_loss_5": (0, round(_lerp(0, 40, p))),
            "end_loss_3": (0, round(_lerp(0, 40, p))),
            "indel_count": self.indel_count(p),
            "seq_error_rate": _lerp(0.0, 0.02, p) * SEQUENCING_ERROR_SCALE,
            "ambiguous_count": (0, round(_lerp(0, 5, p))),
        }

    def stage(self, p: float) -> int:
        """The stage (0 to ``stages - 1``) progress ``p`` falls in."""
        p = max(0.0, min(1.0, p))
        return min(self.stages - 1, int(p * self.stages))

    def describe(self, p: float) -> str:
        """A one-line summary of the settings at progress ``p``."""
        pr = self.params(p)
        indel = (f"indel≤{pr['indel_count'][1]}" if self.indel_form == "uniform"
                 else f"indel {100 * self.indel_share(p):.0f}%")
        return (f"curriculum stage {self.stage(p) + 1}/{self.stages} (p={p:.2f}): "
                f"mut≤{pr['mutation_rate']:.3f}, end_loss≤{pr['end_loss_5'][1]}/{pr['end_loss_3'][1]}, "
                f"{indel}, seq_err≤{pr['seq_error_rate']:.3f}, "
                f"N≤{pr['ambiguous_count'][1]}")
