"""Boundary decoding: a boundary is the argmax of its masked position logits.

The model's soft-argmax expectation lands between the modes of a bimodal distribution; the argmax lands on one. The
expectation stays a model output, but what predict reports is the argmax. The integer position means the same
coordinate the floored expectation of a sharp distribution gave, so ``correct_segments`` consumes it unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from ..core.anchors import codon_starts
from ..errors import NumericalError


def decode_boundaries(start_logits: torch.Tensor, end_logits: torch.Tensor, start_mask: torch.Tensor,
                      end_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per read, ``(start, end)`` as long positions: the argmax of each logit row over its allowed slots.
    A tie takes the first position, on every device. A row with no allowed slot decodes to 0. A non-finite logit on
    an allowed slot is a NumericalError; one on a slot that is not allowed is never decoded."""
    return _masked_argmax(start_logits, start_mask), _masked_argmax(end_logits, end_mask)


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor, what: str = "boundary") -> torch.Tensor:
    """The first allowed slot holding the row's largest logit; ``what`` the logits are names them in the error."""
    if not torch.isfinite(logits[mask]).all():
        raise NumericalError(f"non-finite {what} logit on an allowed slot", context={"field": what})
    logits = logits.masked_fill(~mask, float("-inf"))
    positions = torch.arange(logits.shape[-1], device=logits.device).expand_as(logits)
    return torch.where(logits == logits.max(-1, keepdim=True).values, positions, logits.shape[-1]).min(-1).values


def decode_anchors(logits: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
    """Per read, an anchor's position (long): the argmax of its logits (B, L + 1) over the codon starts of the read's
    positions ``read`` (B, L; ``core.anchors.codon_starts``: where a whole codon fits, as the anchor loss trains) and
    the null slot L, which means the anchor is not observed. A tie takes the first slot; a non-finite logit on an
    allowed slot is a NumericalError."""
    starts = codon_starts(_bool(read))
    return _masked_argmax(torch.as_tensor(logits), torch.cat([starts, starts.new_ones(starts.shape[0], 1)], dim=-1),
                          what="anchor")


def boundary_slots(outputs: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    """The (start, end) half-open slot masks of a model's outputs (``core.boundaries.boundary_masks``)."""
    return _bool(outputs["start_mask"]), _bool(outputs["end_mask"])


def _bool(mask) -> torch.Tensor:
    return torch.as_tensor(mask).bool()
