"""Compatibility imports for translated-read adaptation."""
from ..data.adapters import (build_protein_allele_batch as build_batch_aa,
                             translated_record_usable as aa_usable)

__all__ = ["aa_usable", "build_batch_aa"]
