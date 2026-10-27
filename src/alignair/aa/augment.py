"""Compatibility imports; protein corruption operators live in data.augmentation."""
from ..data.augmentation.protein import aa_substitute, aa_indel, il_ambiguity, aa_truncate

__all__ = ["aa_substitute", "aa_indel", "il_ambiguity", "aa_truncate"]
