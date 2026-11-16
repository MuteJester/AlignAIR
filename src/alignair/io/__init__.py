"""Sequence-file parsing and output serialization."""
from .metadata import DuplicateMetadataIdError, MetadataIndex, build_metadata_index, load_metadata, lookup_metadata_rows
from .sequence_reader import SequenceRead, SequenceReadBatch, iter_sequence_batches
from .sequence_records import RawSequenceRecord, detect_sequence_format, iter_sequence_records

__all__ = [
    "DuplicateMetadataIdError",
    "MetadataIndex",
    "RawSequenceRecord",
    "SequenceRead",
    "SequenceReadBatch",
    "build_metadata_index",
    "detect_sequence_format",
    "iter_sequence_batches",
    "iter_sequence_records",
    "load_metadata",
    "lookup_metadata_rows",
]
