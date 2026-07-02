"""GymHUD: 8-bit 'climb the tower' terminal view. Pure function of GymState."""
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

    def _floor_row(self, state: GymState) -> str:
        cells = []
        for i in range(state.n_levels):
            if i < state.level:
                cells.append(_c("[✓]", "32", self.color))     # cleared (green)
            elif i == state.level:
                cells.append(_c("[▶]", "33", self.color))     # current (yellow)
            else:
                cells.append("[ ]")
        return "".join(cells)

    def render(self, state: GymState) -> str:
        title = _c(" A L I G N A I R   G Y M ", "1;36", self.color)
        lines = [
            "╔══════════════" + title + "══════════════╗",
            f"  FLOOR {state.level + 1}/{state.n_levels}  \"{state.level_name}\"    step {state.step:,}",
            "  " + self._floor_row(state),
            "",
            "  LOCKS (all must open to climb):",
        ]
        for g in state.gates:
            glyph = _c("🔓", "32", self.color) if g.is_open else _c("🔒", "31", self.color)
            lines.append(f"   {g.name:<11} {_bar(g.fraction)}  "
                         f"{g.value:.3g} / {g.threshold:.3g}  {glyph}")
        if state.headline:
            lines += ["", f"  ⚠ STUCK ON: {state.headline}"]
        lines.append("╚" + "═" * 47 + "╝")
        lines.append(f"   ROOM CLEARED ×{state.rooms_cleared}   ·   "
                     f"patience {state.patience_used}/{state.patience_max}   ·   "
                     f"best floor {state.best_level + 1}")
        return "\n".join(lines)

    def event(self, kind: str, state: GymState) -> str:
        msg = {
            "cleared": f"★ ROOM CLEARED — LEVEL UP! now on floor {state.level + 1} ★",
            "best": f"⚑ NEW BEST FLOOR: {state.best_level + 1}",
            "ceiling": f"☠ CEILING REACHED at floor {state.level + 1} — can't climb further",
            "complete": "✦ GYM COMPLETE — all floors cleared ✦",
            "demoted": f"⚠ REGRESSION DETECTED — DEMOTED! back to floor {state.level + 1} ⚠",
        }.get(kind, kind)
        codes = {"cleared": "1;32", "best": "1;33", "ceiling": "1;31", "complete": "1;35", "demoted": "1;31"}
        return _c(msg, codes.get(kind, "1"), self.color)
