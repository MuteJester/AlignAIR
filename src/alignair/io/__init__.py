"""Reading input reads and writing AIRR tables. ``read_sequences``, ``iter_sequences``, ``write_airr`` and
``AirrWriter`` are on the top-level package; here are the batch reader that keeps each read's source ID, and the
metadata index that joins per-read metadata to the reads by that ID."""
from .metadata import MetadataIndex, build_metadata_index, lookup_metadata_rows
from .sequence_reader import SequenceReadBatch, iter_sequence_batches

__all__ = [
    "MetadataIndex",
    "SequenceReadBatch",
    "build_metadata_index",
    "iter_sequence_batches",
    "lookup_metadata_rows",
]
