"""GymHUD: a terminal view of curriculum-training progress. Pure function of GymState."""
from .state import GymState

_BAR_W = 9


def _c(s: str, code: str, on: bool) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if on else s


def _bar(frac: float) -> str:
    filled = int(round(max(0.0, min(1.0, frac)) * _BAR_W))
    return "█" * filled + "░" * (_BAR_W - filled)


class GymHUD:
    def __init__(self, color: bool = True):
        self.color = color

    def _level_row(self, state: GymState) -> str:
        cells = []
        for i in range(state.n_levels):
            if i < state.level:
                cells.append(_c("[x]", "32", self.color))      # passed (green)
            elif i == state.level:
                cells.append(_c("[>]", "33", self.color))      # current (yellow)
            else:
                cells.append("[ ]")
        return "".join(cells)

    def render(self, state: GymState) -> str:
        title = _c(" AlignAIR training curriculum ", "1;36", self.color)
        lines = [
            "=" * 14 + title + "=" * 14,
            f"  Level {state.level + 1}/{state.n_levels}  \"{state.level_name}\"    step {state.step:,}",
            "  " + self._level_row(state),
            "",
            "  Gates (all must pass to advance):",
        ]
        for g in state.gates:
            mark = _c("pass", "32", self.color) if g.is_open else _c("----", "31", self.color)
            lines.append(f"   {g.name:<11} {_bar(g.fraction)}  "
                         f"{g.value:.3g} / {g.threshold:.3g}  {mark}")
        if state.headline:
            lines += ["", f"  Blocked on: {state.headline}"]
        lines.append("=" * 48)
        lines.append(f"   gate-passes {state.rooms_cleared}   ·   "
                     f"patience {state.patience_used}/{state.patience_max}   ·   "
                     f"best level {state.best_level + 1}")
        return "\n".join(lines)

    def event(self, kind: str, state: GymState) -> str:
        msg = {
            "cleared": f"Advanced to level {state.level + 1}",
            "best": f"New best level: {state.best_level + 1}",
            "ceiling": f"Reached the top level ({state.level + 1}); cannot advance further",
            "complete": "Curriculum complete: all levels passed",
            "demoted": f"Regression detected: dropped back to level {state.level + 1}",
        }.get(kind, kind)
        codes = {"cleared": "1;32", "best": "1;33", "ceiling": "1;31", "complete": "1;35", "demoted": "1;31"}
        return _c(msg, codes.get(kind, "1"), self.color)
