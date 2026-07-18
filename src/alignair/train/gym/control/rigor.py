"""Phase-5 statistical rigor for promotion / ceiling / anti-forgetting."""


def mann_kendall_trend(series, eps: float = 0.05) -> str:
    """Sign of Kendall's tau over the series -> "up" | "flat" | "down".
    tau = S / (n(n-1)/2), S = sum_{i<j} sign(x_j - x_i). |tau| < eps => flat."""
    xs = list(series)
    n = len(xs)
    if n < 2:
        return "flat"
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = xs[j] - xs[i]
            s += (d > 0) - (d < 0)
    tau = s / (n * (n - 1) / 2)
    if tau > eps:
        return "up"
    if tau < -eps:
        return "down"
    return "flat"


class HardenedCeiling:
    """Declares a capacity ceiling only when competence has genuinely stopped rising
    (Mann-Kendall flat/down over the window) AND is at/above the achievable floor. A
    flat trajectory BELOW the floor is a sampler STALL (re-explore), not a ceiling."""

    def __init__(self, window: int = 6, eps: float = 0.05, floor=None):
        self.window = window
        self.eps = eps
        self.floor = floor
        self._hist = []

    def update(self, composite: float) -> str:
        self._hist.append(float(composite))
        if len(self._hist) < self.window:
            return "improving"
        if mann_kendall_trend(self._hist[-self.window:], self.eps) == "up":
            return "improving"
        if self.floor is not None and composite < self.floor:
            return "stall"
        return "ceiling"


class RegressionGuard:
    """Anti-forgetting: flags cells whose lower-confidence-bound competence has dropped
    more than `margin` below its best-seen value (a forgetting alarm). The flagged cells
    feed the Phase-4 regret targeting so the sampler re-concentrates on them."""

    def __init__(self, margin: float = 0.03):
        self.margin = margin
        self._best = {}

    def check(self, field: dict) -> list:
        regressed = []
        for cell, v in field.items():
            lcb = float(v["lo"] if isinstance(v, dict) else v)
            # Use mean score S if available, else fall back to lo or v
            val_for_best = float(v.get("S", v.get("lo", v)) if isinstance(v, dict) else v)
            best = self._best.get(cell)
            if best is not None and lcb < best - self.margin:
                regressed.append(cell)
            self._best[cell] = val_for_best if best is None else max(best, val_for_best)
        return regressed
