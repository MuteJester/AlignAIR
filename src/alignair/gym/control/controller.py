"""GymController: runs competence exams, promotes on all-locks-open, detects the
ceiling on plateau, and drives the curriculum progress. The evaluator is injected
(a callable returning a metrics dict) so the control logic tests without a model."""
from typing import Callable

from .config import GymConfig
from .gate import PromotionGate
from .ladder import RankLadder
from .plateau import PlateauDetector
from .state import GymState, composite_score


def _derive_coords_mae(metrics: dict) -> float | None:
    devs = []
    for g in ("v", "j", "d"):
        for suffix in ("e2e_gl_start_dev", "e2e_gl_end_dev"):
            key = f"{g}_{suffix}"
            if key in metrics:
                devs.append(float(metrics[key]))
    return sum(devs) / len(devs) if devs else None


class GymController:
    def __init__(self, config: GymConfig, ladder: RankLadder, gate: PromotionGate,
                 evaluator: Callable[[int, int], dict], hud=None, emit: Callable = print):
        self.config = config
        self.ladder = ladder
        self.gate = gate
        self.evaluator = evaluator
        self.hud = hud
        self.emit = emit
        self.plateau = PlateauDetector(config.patience, config.slope_eps)
        self.level = 0
        self.best_level = 0
        self.rooms_cleared = 0
        self.done = False

    def progress(self) -> float:
        return self.ladder.progress(self.level)

    def _metrics(self, level: int) -> dict:
        metrics = dict(self.evaluator(level, self.config.exam_batches))
        cm = _derive_coords_mae(metrics)
        if cm is not None:
            metrics.setdefault("coords_mae", cm)
        return metrics

    def _snapshot(self, metrics: dict, step: int, headline: str = "") -> GymState:
        sts = self.gate.statuses(metrics, self.level)
        return GymState(
            level=self.level, level_name=self.ladder.name(self.level),
            n_levels=self.config.n_levels, step=step, gates=tuple(sts), axes=(),
            rooms_cleared=self.rooms_cleared, patience_used=self.plateau.used,
            patience_max=self.config.patience, best_level=self.best_level,
            headline=headline)

    def exam(self, step: int) -> GymState:
        metrics = self._metrics(self.level)
        promote, blocking = self.gate.evaluate(metrics, self.level)
        headline = "" if promote else ", ".join(blocking)
        state = self._snapshot(metrics, step, headline)
        if self.hud is not None:
            self.emit(self.hud.render(state))
        if promote:
            self.plateau.reset()
            if self.level >= self.ladder.top:
                self.done = True
                self._event("complete", state)
            else:
                self.level += 1
                self.rooms_cleared += 1
                if self.level > self.best_level:
                    self.best_level = self.level
                self._event("cleared", self._snapshot(metrics, step))
        else:
            if self.plateau.update(composite_score(state.gates)):
                self.done = True
                self._event("ceiling", state)
        return state

    def _event(self, kind: str, state: GymState) -> None:
        if self.hud is not None:
            self.emit(self.hud.event(kind, state))
