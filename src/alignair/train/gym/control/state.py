"""Immutable snapshots of the gym's competence state — the single source of
truth that both the HUD and the JSON/markdown reports render from."""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GateStatus:
    name: str
    value: float
    threshold: float
    direction: str            # "higher" | "lower"

    @property
    def is_open(self) -> bool:
        if self.direction == "higher":
            return self.value >= self.threshold
        return self.value <= self.threshold

    @property
    def fraction(self) -> float:
        """Progress toward the bar in [0,1] (for the HUD progress bar)."""
        if self.threshold == 0:
            return 1.0 if self.is_open else 0.0
        if self.direction == "higher":
            return max(0.0, min(1.0, self.value / self.threshold))
        # lower-better: closer to (or below) threshold => fuller bar
        if self.value <= 0:
            return 1.0
        return max(0.0, min(1.0, self.threshold / self.value))


@dataclass(frozen=True)
class AxisStat:
    axis: str
    bins: tuple                       # tuple[(bin_label: str, value: float, n: int), ...]


@dataclass(frozen=True)
class GymState:
    level: int
    level_name: str
    n_levels: int
    step: int
    gates: tuple                      # tuple[GateStatus, ...]
    axes: tuple                       # tuple[AxisStat, ...]
    rooms_cleared: int
    patience_used: int
    patience_max: int
    best_level: int
    headline: str

    @property
    def all_open(self) -> bool:
        return all(g.is_open for g in self.gates)

    @property
    def blocking(self) -> tuple:
        return tuple(g.name for g in self.gates if not g.is_open)


def composite_score(gates: Sequence[GateStatus]) -> float:
    gs = list(gates)
    if not gs:
        return 0.0
    return sum(g.fraction for g in gs) / len(gs)
