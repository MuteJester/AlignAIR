"""A model's size by component: its parameters and its Conv1d layers, read from the model's own inspection points
(``AlignAIR.component_modules``) and the units of its core.

``model_size(model)`` gives the total parameters and Conv1d layers, and per component:

- ``embedding`` and ``orientation_head``, when the model has them;
- ``core.<unit>``, one per unit of the core that holds parameters or layers (the chip's ``trunk``, ``parser``,
  ``anchors``, ``reader``, ``scans`` and ``d_presence``), and ``core`` for the core's parameters no unit holds;
- ``unattributed``: the parameters and Conv1d layers no component holds, when there are any.

A parameter or layer shared between components is counted once, in the first that holds it. The components are
summed into ``GROUPS``: trunk (the core's trunk), parser, reader (the SHM reader), scans (the bank scans), heads (the
anchors and the D presence evidence) and other (the rest), which add up to the total. ``size_line`` puts one model's
groups (in millions), Conv1d count and training rate on one line of a screen's headline.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import nn

GROUPS = ("trunk", "parser", "reader", "scans", "heads", "other")
_UNIT_GROUPS = {"core.trunk": "trunk", "core.parser": "parser", "core.reader": "reader", "core.scans": "scans",
                "core.anchors": "heads", "core.d_presence": "heads"}


def _components(model: nn.Module) -> dict[str, nn.Module]:
    """The components, in counting order: the inspection points, the core's units before the core itself."""
    points = dict(model.component_modules())
    core = points.pop("core", None)
    if core is not None:
        points.update({f"core.{name}": unit for name, unit in core.named_children() if unit is not None})
        points["core"] = core
    return points


def model_size(model: nn.Module) -> dict:
    """``{"parameters", "conv1d", "groups", "components"}`` of an AlignAIR model (see the module docstring); each
    component is ``{"parameters", "conv1d"}``."""
    seen_parameters: set[int] = set()
    seen_layers: set[int] = set()
    components = {}
    for name, module in _components(model).items():
        parameters = [p for p in module.parameters() if id(p) not in seen_parameters]
        layers = [m for m in module.modules() if isinstance(m, nn.Conv1d) and id(m) not in seen_layers]
        seen_parameters.update(id(p) for p in parameters)
        seen_layers.update(id(m) for m in layers)
        if parameters or layers or not name.startswith("core"):      # an empty core unit (the bound bank) is left out
            components[name] = {"parameters": sum(p.numel() for p in parameters), "conv1d": len(layers)}
    total = sum(p.numel() for p in model.parameters())
    conv1d = sum(isinstance(m, nn.Conv1d) for m in model.modules())
    rest = {"parameters": total - sum(component["parameters"] for component in components.values()),
            "conv1d": conv1d - sum(component["conv1d"] for component in components.values())}
    if any(rest.values()):
        components["unattributed"] = rest
    groups = dict.fromkeys(GROUPS, 0)
    for name, component in components.items():
        groups[_UNIT_GROUPS.get(name, "other")] += component["parameters"]
    return {"parameters": total, "conv1d": conv1d, "groups": groups, "components": components}


def checkpoint_size(path: str | Path) -> dict:
    """``model_size`` of the model in the ``.alignair`` file at ``path``, with its ``architecture``. A lab candidate's
    file is read after this process registered the candidates."""
    import alignair_lab.architectures  # noqa: F401  (a candidate's model file names a lab architecture)
    from alignair.model_file import load_model
    model = load_model(str(path), device="cpu").model
    return {**model_size(model), "architecture": model.cfg.architecture}


def size_line(label: str, size: Mapping, steps_per_second: float | None) -> str:
    """One model's size on one line: its total and its non-empty groups in millions, its Conv1d count and its training
    rate (``steps_per_second``; None when not recorded)."""
    groups = size["groups"]
    shown = [group for group in GROUPS if groups[group]]
    parts = ", ".join(f"{group} {groups[group] / 1e6:.3f}M" for group in shown)
    rate = "steps/s not recorded" if steps_per_second is None else f"{steps_per_second} steps/s"
    return (f"Size of the {label} ({size['architecture']}): {size['parameters'] / 1e6:.3f}M parameters ({parts}), "
            f"{size['conv1d']} Conv1d, {rate}.")
