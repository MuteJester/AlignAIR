"""Boundary-slot masks shared by every model using the gene-branch interface."""
from __future__ import annotations

import torch

from ..errors import ConfigurationError


def read_window(max_seq_length: int, boundary_mode: str = "residue") -> int:
    """The longest read a model sees: the padded width, less the terminal slot a half-open model reserves for a
    segment end at the read's length."""
    return max_seq_length - int(boundary_mode == "half_open")


def boundary_masks(tokens: torch.Tensor, mode: str = "half_open") -> tuple[torch.Tensor, torch.Tensor]:
    """Return start/end masks, with True marking an allowed boundary slot.

    ``residue`` preserves the released nucleotide decoder's mask for both boundaries.
    ``half_open`` allows end slot k iff token k-1 is present, so a read of length n can
    end at n. That model needs a spare slot: its encoded read must be shorter than the window.
    """
    valid = tokens != 0
    if mode == "residue":
        return valid, valid
    if mode != "half_open":
        raise ConfigurationError(f"unsupported boundary_mode: {mode!r}")
    end = torch.zeros_like(valid)
    end[:, 1:] = valid[:, :-1]
    return valid, end
