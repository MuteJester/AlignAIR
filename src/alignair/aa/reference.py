"""Compatibility imports for shared protein reference views."""
from ..reference.protein import (
    DEFAULT_ANCHOR_RESIDUES as ANCHOR_AA,
    ProteinGermline as AAGermline,
    ProteinGermlineSet as AAGermlineSet,
    translate_germline,
    build_protein_germlines as build_aa_germlines,
)

__all__ = ["ANCHOR_AA", "AAGermline", "AAGermlineSet", "translate_germline", "build_aa_germlines"]
