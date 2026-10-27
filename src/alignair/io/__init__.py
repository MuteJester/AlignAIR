"""Sequence-file parsing and output serialization."""
from .sequence_records import RawSequenceRecord, detect_sequence_format, iter_sequence_records

__all__ = ["RawSequenceRecord", "detect_sequence_format", "iter_sequence_records"]
