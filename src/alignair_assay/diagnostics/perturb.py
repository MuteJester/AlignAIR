"""A deterministically perturbed copy of an AlignAIR model file: the positive control of the readiness audit.

``zero_module(checkpoint, out, module)`` writes ``out``: the ``.alignair`` file ``checkpoint`` with every parameter
of one module of the model (a dotted name, e.g. ``core.parser``) set to zero, and every other byte of the
file as it was: the header and every other section are re-emitted unchanged (as ``alignair.model_file.update_card``
re-emits them), only the ``weights`` section is rewritten. The same input gives the same bytes, so the perturbed model
has one weights digest, and its model key changes only with the input or the module.
"""
from __future__ import annotations

from pathlib import Path

import torch
from alignair.model_file import container, serialize


def zero_module(checkpoint, out, module: str) -> dict:
    """Write ``out`` as the module docstring says; returns ``module``, the zeroed ``parameters`` (sorted by name:
    safetensors hands the weights back in no fixed order) and how many ``values`` they hold. ValueError, before
    anything is written, when the weights hold no parameter of ``module``."""
    checkpoint, out = str(Path(checkpoint).expanduser()), Path(out)
    header = container.read_header(checkpoint)
    sections, formats = {}, {}
    for name, section in header["sections"].items():
        sections[name] = (container.read_section(checkpoint, name), section["codec"])  # verifies the checksums
        formats[name] = section.get("format", "bytes")
    weights = serialize.state_dict_from_bytes(sections["weights"][0])
    names = sorted(name for name in weights if name.startswith(f"{module}."))
    if not names:
        raise ValueError(f"{checkpoint} has no parameter of module {module!r} to perturb")
    for name in names:
        weights[name] = torch.zeros_like(weights[name])
    sections["weights"] = (serialize.state_dict_to_bytes(weights), sections["weights"][1])
    kept = {key: value for key, value in header.items() if not key.startswith("_") and key != "sections"}
    out.parent.mkdir(parents=True, exist_ok=True)
    container.write_container(str(out), {**kept, "_formats": formats}, sections)
    return {"module": module, "parameters": names, "values": sum(weights[name].numel() for name in names)}
