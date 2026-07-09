"""Compatibility exports for built-in benchmark generation recipes."""
from __future__ import annotations

from .recipes import (
    adaptive_igh_strata,
    default_igh_assay_spec,
    default_igh_spec,
    focused_igh_spec,
    focused_igh_strata,
    isolated_params,
)

__all__ = [
    "adaptive_igh_strata",
    "default_igh_assay_spec",
    "default_igh_spec",
    "focused_igh_spec",
    "focused_igh_strata",
    "isolated_params",
]
