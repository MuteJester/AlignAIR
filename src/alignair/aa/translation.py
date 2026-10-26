"""Compatibility imports for shared translation and codon-coordinate projection."""
from ..data.projection import AAProjection, CODON_TABLE, frame_offset_of, project_codons, translate_nt

__all__ = ["AAProjection", "CODON_TABLE", "frame_offset_of", "project_codons", "translate_nt"]
