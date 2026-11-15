"""Sequence-file parsing and output serialization."""
from .sequence_records import RawSequenceRecord, detect_sequence_format, iter_sequence_records
from .metadata import DuplicateMetadataIdError, MetadataIndex, build_metadata_index, load_metadata, lookup_metadata_rows
from .sequence_reader import SequenceRead, SequenceReadBatch, iter_sequence_batches

__all__ = ["RawSequenceRecord", "detect_sequence_format", "iter_sequence_records",
           "DuplicateMetadataIdError", "MetadataIndex", "build_metadata_index", "load_metadata", "lookup_metadata_rows",
           "SequenceRead", "SequenceReadBatch", "iter_sequence_batches"]
