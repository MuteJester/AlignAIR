"""Shape-matched warm starts from another model's weights.

Every parameter and buffer whose name and shape match the source's is copied; the rest keep the target's fresh
initialization. A chip's weights do not depend on its reference (the alleles are an input, the bound bank), so a
source of the same widths (``embed_dim``, ``trunk_channels``) transfers whole; a part whose shape differs (another
width, or a gene the target does not have) does not. A match of names and shapes is not a guarantee of architectural
compatibility.
"""
from __future__ import annotations

import collections


def transfer_compatible_weights(target_model, source_state_dict: dict) -> tuple[list, list]:
    """Copy shape-matching tensors from ``source_state_dict`` into ``target_model`` (in place).

    Returns ``(transferred, skipped)`` state-dict key lists. A key is transferred iff the source holds
    the same name with an identical shape; every other target key keeps its current (fresh) value."""
    tgt = target_model.state_dict()
    merged, transferred, skipped = {}, [], []
    for k, v in tgt.items():
        s = source_state_dict.get(k)
        if s is not None and tuple(s.shape) == tuple(v.shape):
            merged[k] = s
            transferred.append(k)
        else:
            merged[k] = v
            skipped.append(k)
    target_model.load_state_dict(merged)
    return transferred, skipped


def load_source_state_dict(source_path: str, device: str = "cpu") -> dict:
    """State_dict of another AlignAIR model (any reference/architecture), for backbone transfer."""
    from ..model_file import read_model_state
    state = read_model_state(source_path).model_state
    return {name: value.to(device) for name, value in state.items()}


def summarize(transferred: list, skipped: list, depth: int = 2) -> str:
    """One-line-per-module-group summary of what transferred vs skipped (grouped by name prefix)."""
    def group(keys):
        c = collections.Counter(".".join(k.split(".")[:depth]) for k in keys)
        return ", ".join(f"{g}({n})" for g, n in sorted(c.items()))
    n = len(transferred) + len(skipped)
    pct = 100 * len(transferred) / n if n else 0.0
    lines = [f"backbone transfer: {len(transferred)}/{n} tensors ({pct:.0f}%)",
             f"  transferred: {group(transferred)}",
             f"  skipped (fresh init): {group(skipped)}"]
    return "\n".join(lines)
