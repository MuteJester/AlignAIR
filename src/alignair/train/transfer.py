"""Backbone transfer: warm-start a model on a NEW reference from another model's weights.

The shared, reference-agnostic backbone — token/position embedding, orientation head, the shared meta
tower, per-gene segmentation towers + boundary heads, and the mutation/indel/productivity meta-heads —
has the same shape across references (it depends on ``embed_dim``/``filters``/``block_out``/``max_len``,
not on the allele counts). It transfers directly. The reference-specific parts — allele classification
heads, an auto-scaled classification tower, a chain_type head, or a gene absent in the new locus — differ
in shape and keep their fresh initialization.

Transfer is a shape-matched partial load: copy every parameter/buffer whose NAME **and** SHAPE match the
source; leave the rest at the target's init. General across any source→target pair.
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


def load_source_state_dict(source_path: str, device: str = "cpu", trust_pickle: bool = False) -> dict:
    """State_dict of another AlignAIR model (any reference/architecture), for backbone transfer."""
    from ..api import load_model
    src, _ = load_model(source_path, device=device, trust_pickle=trust_pickle)
    return src.state_dict()


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
