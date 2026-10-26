"""Compatibility imports for the shared configurable AlignAIR implementation.

New model construction can use alignair.core.AlignAIR for either alphabet. These constructors
retain the established protein-class preset for existing studies and bundles.
"""
from __future__ import annotations

from ..core.boundaries import boundary_masks
from ..core.config import AlignAIRConfig
from ..core.gene_branch import GeneBranch
from ..core.model import AlignAIR

__all__ = ["AAAlignAIR", "AAGeneBranch", "boundary_masks"]


class AAGeneBranch(GeneBranch):
    """Compatibility constructor selecting categorical classification."""

    def __init__(self, spec, cfg: AlignAIRConfig):
        super().__init__(spec, cfg.with_amino_acid_heads())


class AAAlignAIR(AlignAIR):
    """Compatibility constructor for the established protein-class model preset."""

    def __init__(self, cfg: AlignAIRConfig):
        super().__init__(cfg.with_amino_acid_heads())
