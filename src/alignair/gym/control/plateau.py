"""PlateauDetector: declares a capacity ceiling when the composite competence
score stops improving for `patience` consecutive exams."""


class PlateauDetector:
    def __init__(self, patience: int = 8, slope_eps: float = 1e-3):
        self.patience = patience
        self.slope_eps = slope_eps
        self._best = None
        self._stall = 0

    @property
    def used(self) -> int:
        return self._stall

    def reset(self) -> None:
        self._best = None
        self._stall = 0

    def update(self, composite: float) -> bool:
        if self._best is None or composite > self._best + self.slope_eps:
            self._best = composite
            self._stall = 0
            return False
        self._stall += 1
        return self._stall >= self.patience
