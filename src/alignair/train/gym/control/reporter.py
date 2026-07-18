"""StruggleReporter: render a GymState to JSON + markdown + a progress-curve line."""
import json
import os

from .state import GymState, composite_score


class StruggleReporter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def to_dict(self, state: GymState) -> dict:
        return {
            "level": state.level,
            "level_name": state.level_name,
            "n_levels": state.n_levels,
            "step": state.step,
            "composite": composite_score(state.gates),
            "blocking": list(state.blocking),
            "headline": state.headline,
            "rooms_cleared": state.rooms_cleared,
            "best_level": state.best_level,
            "gates": [{"name": g.name, "value": g.value, "threshold": g.threshold,
                       "direction": g.direction, "open": g.is_open} for g in state.gates],
            "axes": [{"axis": a.axis,
                      "bins": [{"label": b[0], "value": b[1], "n": b[2]} for b in a.bins]}
                     for a in state.axes],
        }

    def markdown(self, state: GymState) -> str:
        d = self.to_dict(state)
        lines = [f"# Curriculum report — level {state.level + 1}/{state.n_levels} "
                 f"\"{state.level_name}\" @ step {state.step:,}",
                 "", f"**Composite:** {d['composite']:.3f}  ·  "
                 f"**Blocking:** {', '.join(d['blocking']) or 'none (all passed)'}", "",
                 "## Gates", "", "| gate | value | threshold | open |",
                 "|---|---|---|---|"]
        for g in d["gates"]:
            lines.append(f"| {g['name']} | {g['value']:.3g} | {g['threshold']:.3g} "
                         f"| {'yes' if g['open'] else 'no'} |")
        for a in d["axes"]:
            lines += ["", f"## Axis: {a['axis']}", "", "| bin | metric | n |", "|---|---|---|"]
            for b in a["bins"]:
                lines.append(f"| {b['label']} | {b['value']:.3f} | {b['n']} |")
        return "\n".join(lines)

    def write(self, state: GymState) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        base = f"gym_report_floor{state.level}_step{state.step}"
        jpath = os.path.join(self.out_dir, base + ".json")
        with open(jpath, "w") as f:
            json.dump(self.to_dict(state), f, indent=2)
        with open(os.path.join(self.out_dir, base + ".md"), "w") as f:
            f.write(self.markdown(state))
        with open(os.path.join(self.out_dir, "climb_curve.jsonl"), "a") as f:
            f.write(json.dumps({"step": state.step, "level": state.level,
                                "composite": composite_score(state.gates),
                                "best_level": state.best_level}) + "\n")
        return jpath
