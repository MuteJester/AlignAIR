"""Shared GenAIRR parameter backgrounds for benchmark recipes."""
from __future__ import annotations

_CLEAN_BACKGROUND = {
    "mutation_rate": 0.005,
    "end_loss_5": (0, 0),
    "end_loss_3": (0, 0),
    "indel_count": (0, 0),
    "seq_error_rate": 0.0,
    "ambiguous_count": (0, 0),
    "crop_prob": 0.0,
    "invert_d_prob": 0.0,
}


def isolated_params(**overrides) -> dict:
    """Return clean-background GenAIRR params plus explicit overrides."""

    params = dict(_CLEAN_BACKGROUND)
    params.update(overrides)
    return params
