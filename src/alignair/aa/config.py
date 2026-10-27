"""Compatibility factory for the shared protein-class configuration policy."""
from __future__ import annotations

from ..core.setup import build_model_config
from ..core.config import AlignAIRConfig


def aa_igh_config(n_v_classes: int, n_j_classes: int, n_d_alleles: int = 0,
                  max_seq_length: int = 192, has_d: bool = False) -> AlignAIRConfig:
    return build_model_config(n_v_classes, n_j_classes, n_d_alleles, input_space="protein",
                              label_space="protein_classes", has_d=has_d, max_seq_length=max_seq_length)
