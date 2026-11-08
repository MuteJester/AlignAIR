"""Half-open boundary slots: a segment of a read starts at a residue and ends one past its last residue."""
from __future__ import annotations

import torch


# A forward's width in evaluation is rounded up to a multiple of this (``width_bucket``): reads predicted together are
# grouped by their bucket, so a read's width, and its outputs, depend on the read alone (``predict.forward.run_model``).
WIDTH_BUCKET = 32


def width_bucket(n: int, whole: int) -> int:
    """``n`` positions rounded up to a multiple of ``WIDTH_BUCKET``, at most ``whole`` (the padded window)."""
    return min(whole, -(-n // WIDTH_BUCKET) * WIDTH_BUCKET)


def read_window(max_seq_length: int) -> int:
    """The longest read a model sees: the padded width, less the terminal slot reserved for a segment end at the
    read's length."""
    return max_seq_length - 1


def boundary_masks(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return start/end masks, with True marking an allowed boundary slot: a start on any read position, end slot k
    iff token k-1 is present, so a read of length n can end at n (the encoded read is shorter than the window)."""
    valid = tokens != 0
    end = torch.zeros_like(valid)
    end[:, 1:] = valid[:, :-1]
    return valid, end
