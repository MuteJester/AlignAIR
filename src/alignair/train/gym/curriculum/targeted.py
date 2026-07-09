"""TargetedCurriculum and ProgressTracker implementations."""


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


class TargetedCurriculum:
    """Wraps a FactoredCurriculum and emits a MIXTURE of weighted difficulty components."""

    def __init__(self, factored, lattice, tracker=None, p_ramp=0.6, p_alp=0.25,
                 p_floor=0.15, floor_cell="heavy_shm_fulllen"):
        self.factored = factored
        self.lattice = lattice
        self.tracker = tracker or ProgressTracker()
        self.p_ramp, self.p_alp, self.p_floor = p_ramp, p_alp, p_floor
        self.floor_cell = floor_cell
        self._cell_params = {c.name: lattice.cell_params(c) for c in lattice.cells}

    def params(self, p=None):
        return self.factored.params(p)

    def describe(self, p=None):
        return "targeted | " + self.factored.describe(p)

    def stage(self, p=None):
        return self.factored.stage(p)

    @property
    def pace(self):
        return self.factored.pace

    @property
    def axes(self):
        return self.factored.axes

    def advance(self, axis_competence, **kw):
        return self.factored.advance(axis_competence, **kw)

    def update_targets(self, field: dict) -> None:
        self.tracker.update(field)

    def components(self):
        floor = (self.p_floor, self._cell_params.get(
            self.floor_cell, self.factored.params()))
        top = self.tracker.top_cell()
        if top is None or top not in self._cell_params:
            # no targets yet: the ramp absorbs the targeting mass
            return [(self.p_ramp + self.p_alp, self.factored.params()), floor]
        return [(self.p_ramp, self.factored.params()),
                (self.p_alp, self._cell_params[top]), floor]
