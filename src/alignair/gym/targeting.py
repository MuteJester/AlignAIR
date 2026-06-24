"""Phase-4 automatic targeting: turn the lattice competence field over time into
per-cell priorities (Absolute Learning Progress + regret) and express the curriculum
as a mixture of weighted difficulty components (ramp + targeted cell + hard-corner)."""


class ProgressTracker:
    def __init__(self, target: float = 0.95, alp_weight: float = 1.0,
                 regret_weight: float = 0.5):
        self.target = target
        self.alp_weight = alp_weight
        self.regret_weight = regret_weight
        self._prev: dict = {}
        self._alp: dict = {}

    def update(self, field: dict) -> None:
        for cell, v in field.items():
            s = float(v["S"] if isinstance(v, dict) else v)
            if cell in self._prev:
                self._alp[cell] = abs(s - self._prev[cell])
            self._prev[cell] = s

    def _raw(self) -> dict:
        out = {}
        for c, s in self._prev.items():
            alp = self._alp.get(c, 0.0)
            regret = max(0.0, self.target - s)
            out[c] = self.alp_weight * alp + self.regret_weight * regret
        return out

    def priorities(self) -> dict:
        raw = self._raw()
        if not raw:
            return {}
        tot = sum(raw.values())
        if tot <= 0:
            return {c: 1.0 / len(raw) for c in raw}
        return {c: w / tot for c, w in raw.items()}

    def top_cell(self):
        raw = self._raw()
        return max(raw, key=raw.get) if raw else None
