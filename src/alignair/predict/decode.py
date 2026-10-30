"""Boundary decoding: a boundary is the argmax of its masked position logits.

The model's soft-argmax expectation lands between the modes of a bimodal distribution; the argmax lands on one.
Training still needs the expectation (the differentiable soft cutout), so it stays a model output, but what predict
reports is the argmax. The integer position means the same coordinate the floored expectation of a sharp
distribution gave, so ``correct_segments`` consumes it unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from ..errors import NumericalError


def decode_boundaries(start_logits: torch.Tensor, end_logits: torch.Tensor, start_mask: torch.Tensor,
                      end_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per read, ``(start, end)`` as long positions: the argmax of each logit row over its allowed slots.
    A tie takes the first position, on every device. A row with no allowed slot decodes to 0. A non-finite logit on
    an allowed slot is a NumericalError; one on a slot that is not allowed is never decoded."""
    return _masked_argmax(start_logits, start_mask), _masked_argmax(end_logits, end_mask)


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(logits[mask]).all():
        raise NumericalError("non-finite boundary logit on an allowed slot")
    logits = logits.masked_fill(~mask, float("-inf"))
    positions = torch.arange(logits.shape[-1], device=logits.device).expand_as(logits)
    return torch.where(logits == logits.max(-1, keepdim=True).values, positions, logits.shape[-1]).min(-1).values


def decode_anchors(logits: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
    """Per read, an anchor's position (long): the argmax of its logits (B, L + 1) over the read's positions ``read``
    (B, L) and the null slot L, which means the anchor is not observed. A tie takes the first slot; a non-finite logit
    on an allowed slot is a NumericalError."""
    read = _bool(read)
    return _masked_argmax(torch.as_tensor(logits), torch.cat([read, read.new_ones(read.shape[0], 1)], dim=-1))


def decoded_anchors(outputs: Mapping) -> dict[str, torch.Tensor] | None:
    """The V and J anchors of a model's outputs (``decode_anchors``; L for not observed) as ``{"v": ..., "j": ...}``,
    or None when the outputs carry no anchor logits. The read is the start slots: an anchor is a codon start."""
    if not all(f"{gene}_anchor_logits" in outputs for gene in ("v", "j")):
        return None
    read, _ = boundary_slots(outputs)
    return {gene: decode_anchors(outputs[f"{gene}_anchor_logits"], read) for gene in ("v", "j")}


def boundary_slots(outputs: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    """The (start, end) slot masks of a model's outputs. A half-open model gives both. A residue model gives the read's
    bases for the start, and for the end-exclusive end those bases plus the slot after the last one, where a segment
    that runs to the read end stops. That slot is unsupervised in residue mode: the end cross-entropy masks it
    (``tasks/losses.py``, ``tasks/functional.py``), so its logit there is extrapolated from the neighbouring slots,
    not trained. A half-open model supervises its end slot n."""
    if "start_mask" in outputs and "end_mask" in outputs:
        return _bool(outputs["start_mask"]), _bool(outputs["end_mask"])
    valid = _bool(outputs["position_mask"])
    after = torch.zeros_like(valid)
    after[:, 1:] = valid[:, :-1]
    return valid, valid | after


def decoded_boundaries(outputs: Mapping, gene: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One gene's reported ``(start, end)`` from a model's outputs (tensors or arrays), or None when the outputs
    carry no logits or masks to decode (the caller keeps the model's expectations)."""
    keys = (f"{gene}_start_logits", f"{gene}_end_logits")
    if not all(key in outputs for key in keys) or not (
            "position_mask" in outputs or {"start_mask", "end_mask"} <= outputs.keys()):
        return None
    start_slots, end_slots = boundary_slots(outputs)
    return decode_boundaries(torch.as_tensor(outputs[keys[0]]), torch.as_tensor(outputs[keys[1]]),
                             start_slots, end_slots)


def _bool(mask) -> torch.Tensor:
    return torch.as_tensor(mask).bool()
