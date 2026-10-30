"""A model's size by component: its parameters and its Conv1d layers, read from the model's own inspection points
(``AlignAIR.component_modules``, which holds each gene branch's own ``component_modules``).

``model_size(model)`` gives the total parameters and Conv1d layers, and per component:

- ``embedding``, ``orientation_head``, ``trunk``, ``meta_tower`` and ``state_branch``, when the model has them;
- ``meta_heads.<name>`` and ``trunk_heads.<name>``, one per head;
- per gene ``<g>.segmentation`` (its ``SEGMENTATION`` points: a segmentation tower and two boundary heads, or a
  boundary convolution) and ``<g>.classification`` (its ``CLASSIFICATION`` points: the soft cutout, the
  classification tower, its mid layer and head); a branch's other inspection point keeps its own name, ``<g>.<point>``;
- ``unattributed``: the parameters and Conv1d layers no inspection point holds, when there are any.

A parameter or layer shared between components is counted once, in the first that holds it. The components are
summed into ``GROUPS``: segmentation (the genes'), trunk (the trunk and its heads), classification (the genes'), meta
(the meta tower and its heads) and other (the rest), which add up to the total. ``size_line`` puts one model's
groups (in millions), Conv1d count and training rate on one line of a screen's headline.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import nn

SEGMENTATION = ("seg_tower", "start_head", "end_head", "boundary")
CLASSIFICATION = ("cutout", "cls_tower", "cls_mid", "cls_head")
GROUPS = ("segmentation", "trunk", "classification", "meta", "other")


def model_size(model: nn.Module) -> dict:
    """``{"parameters", "conv1d", "groups", "components"}`` of an AlignAIR model (see the module docstring); each
    component is ``{"parameters", "conv1d"}``."""
    genes = set(model.branches)
    held: dict[str, list[nn.Module]] = {}
    for name, module in model.component_modules().items():
        gene, _, point = name.partition(".")
        if gene in genes and point:
            part = ("segmentation" if point in SEGMENTATION else "classification" if point in CLASSIFICATION
                    else point)
            name = f"{gene}.{part}"
        held.setdefault(name, []).append(module)
    seen_parameters: set[int] = set()
    seen_layers: set[int] = set()
    components = {}
    for name, modules in held.items():
        parameters = [p for module in modules for p in module.parameters() if id(p) not in seen_parameters]
        layers = [m for module in modules for m in module.modules()
                  if isinstance(m, nn.Conv1d) and id(m) not in seen_layers]
        seen_parameters.update(id(p) for p in parameters)
        seen_layers.update(id(m) for m in layers)
        components[name] = {"parameters": sum(p.numel() for p in parameters), "conv1d": len(layers)}
    total = sum(p.numel() for p in model.parameters())
    conv1d = sum(isinstance(m, nn.Conv1d) for m in model.modules())
    rest = {"parameters": total - sum(component["parameters"] for component in components.values()),
            "conv1d": conv1d - sum(component["conv1d"] for component in components.values())}
    if any(rest.values()):
        components["unattributed"] = rest
    groups = dict.fromkeys(GROUPS, 0)
    for name, component in components.items():
        groups[_group(name)] += component["parameters"]
    return {"parameters": total, "conv1d": conv1d, "groups": groups, "components": components}


def _group(component: str) -> str:
    if component.endswith(".segmentation"):
        return "segmentation"
    if component.endswith(".classification"):
        return "classification"
    if component == "trunk" or component.startswith("trunk_heads."):
        return "trunk"
    if component == "meta_tower" or component.startswith("meta_heads."):
        return "meta"
    return "other"


def checkpoint_size(path: str | Path) -> dict:
    """``model_size`` of the model in the ``.alignair`` file at ``path``, with its ``architecture``. A lab candidate's
    file is read after this process registered the candidates."""
    import alignair_lab.architectures  # noqa: F401  (a candidate's model file names a lab architecture)
    from alignair.model_file import load_model
    model = load_model(str(path), device="cpu").model
    return {**model_size(model), "architecture": model.cfg.architecture}


def size_line(label: str, size: Mapping, steps_per_second: float | None) -> str:
    """One model's size on one line: its total and groups in millions (the trunk only when it has one), its Conv1d
    count and its training rate (``steps_per_second``; None when not recorded)."""
    groups = size["groups"]
    shown = [group for group in GROUPS if group != "trunk" or groups[group]]
    parts = ", ".join(f"{group} {groups[group] / 1e6:.3f}M" for group in shown)
    rate = "steps/s not recorded" if steps_per_second is None else f"{steps_per_second} steps/s"
    return (f"Size of the {label} ({size['architecture']}): {size['parameters'] / 1e6:.3f}M parameters ({parts}), "
            f"{size['conv1d']} Conv1d, {rate}.")
