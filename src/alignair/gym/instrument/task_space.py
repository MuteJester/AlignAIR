"""TaskSpace: the difficulty parameter box Θ (one axis per GenAIRR knob), with
deployment-99th-percentile maxes so the eval/terminal distribution covers the hard
tail. Maps a sampled θ to the params dict gym.build_experiment consumes."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    name: str
    lo: float
    hi: float
    kind: str          # "rate" | "count" | "prob" | "len"


# crop_len is expressed as the MIN length of the (junction-centered) window; harder =
# shorter, so its "value" is the shortest allowed window. We invert at param time.
_DEPLOY_AXES = (
    Axis("mutation_rate", 0.005, 0.30, "rate"),    # hard tail well past 0.15
    Axis("end_loss_5", 0.0, 120.0, "count"),
    Axis("end_loss_3", 0.0, 45.0, "count"),
    Axis("indel_count", 0.0, 5.0, "count"),
    Axis("seq_error_rate", 0.0, 0.02, "rate"),
    Axis("ambiguous_count", 0.0, 10.0, "count"),
    Axis("crop_len", 50.0, 576.0, "len"),          # shortest junction window allowed
    Axis("orient_prob", 0.0, 0.5, "prob"),
)


class TaskSpace:
    def __init__(self, axes):
        self.axes = tuple(axes)

    @classmethod
    def deployment(cls):
        return cls(_DEPLOY_AXES)

    def sample(self, rng=None, frac: dict | None = None) -> dict:
        """A difficulty point. Axes named in `frac` take that fraction of their range;
        every UNFIXED axis takes its easy BASELINE (lo) so a cell isolates only the
        difficulty it names (uncontrolled axes would confound the measurement and vary
        with seed). `rng` is accepted for API stability but unused (the point is
        deterministic); per-read variation comes from GenAIRR, not from this point."""
        out = {}
        for ax in self.axes:
            if frac is not None and ax.name in frac:
                f = max(0.0, min(1.0, frac[ax.name]))
                out[ax.name] = ax.lo + (ax.hi - ax.lo) * f
            else:
                out[ax.name] = ax.lo
        return out

    def to_genairr_params(self, theta: dict) -> dict:
        def _ct(v):     # count axis -> (0, n) GenAIRR length-range tuple
            return (0, int(round(v)))
        crop_min = int(round(theta["crop_len"]))
        # below the full read length => some reads cropped to a window >= crop_min
        cropped = crop_min < 576
        return {
            "mutation_rate": float(theta["mutation_rate"]),
            "end_loss_5": _ct(theta["end_loss_5"]),
            "end_loss_3": _ct(theta["end_loss_3"]),
            "indel_count": _ct(theta["indel_count"]),
            "seq_error_rate": float(theta["seq_error_rate"]),
            "ambiguous_count": _ct(theta["ambiguous_count"]),
            "crop_prob": 0.6 if cropped else 0.0,
            "crop_len_min": crop_min,
            "crop_len_max": 576 if not cropped else crop_min + 1,
            "orient_prob": float(theta["orient_prob"]),
        }
