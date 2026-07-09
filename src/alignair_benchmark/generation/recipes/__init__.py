"""Built-in GenAIRR-backed benchmark generation recipes."""
from __future__ import annotations

from .adaptive import adaptive_igh_strata
from .background import isolated_params
from .focused import focused_igh_spec, focused_igh_strata
from .igh import default_igh_assay_spec, default_igh_spec

__all__ = [
    "adaptive_igh_strata",
    "default_igh_assay_spec",
    "default_igh_spec",
    "focused_igh_spec",
    "focused_igh_strata",
    "isolated_params",
]
