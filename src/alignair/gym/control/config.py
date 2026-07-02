"""Configuration for the competence-gated gym: gate specs + global knobs."""
from dataclasses import dataclass, field


def _linspace(a: float, b: float, n: int) -> tuple[float, ...]:
    if n == 1:
        return (a,)
    step = (b - a) / (n - 1)
    return tuple(a + step * i for i in range(n))


@dataclass(frozen=True)
class GateSpec:
    """One promotion lock: a metric, whether higher or lower is better, and a
    per-level threshold (thresholds[level]). All gates must open to climb."""
    metric: str
    direction: str            # "higher" | "lower"
    thresholds: tuple[float, ...]

    def __post_init__(self):
        if self.direction not in ("higher", "lower"):
            raise ValueError(f"direction must be higher|lower, got {self.direction}")


def default_gates(n_levels: int) -> tuple[GateSpec, ...]:
    """Relaxing per-level bars: higher-better metrics get a LOWER bar at harder
    levels (they're achievable-ceiling-limited); the coords MAE gate ALLOWS more
    error. Numbers are an initial cut to be recalibrated against a real climb."""
    return (
        GateSpec("v_call", "higher", _linspace(0.97, 0.80, n_levels)),
        GateSpec("d_call", "higher", _linspace(0.85, 0.55, n_levels)),
        GateSpec("j_call", "higher", _linspace(0.92, 0.70, n_levels)),
        GateSpec("coords_mae", "lower", _linspace(1.0, 4.0, n_levels)),
        GateSpec("region_acc", "higher", _linspace(0.99, 0.90, n_levels)),
    )


@dataclass(frozen=True)
class GymConfig:
    n_levels: int = 10
    gates: tuple[GateSpec, ...] = field(default_factory=lambda: default_gates(10))
    exam_every: int = 500          # training steps between competence exams
    exam_batches: int = 8          # eval batches per exam
    patience: int = 8              # exams w/o composite improvement -> ceiling
    slope_eps: float = 1e-3        # min composite gain that counts as progress
    color: bool = True             # HUD ANSI color (auto-off on non-TTY by caller)
    demote_margin: float = 0.08    # absolute drop below promotion gate threshold that triggers a demotion
