"""Reading input reads from FASTA, FASTQ, CSV/TSV (a sequence column) or plain one-per-line TXT, optionally
gzip-compressed. ``iter_sequence_batches`` yields ``SequenceReadBatch``es whose reads are uppercased and validated:
IUPAC ambiguity codes -> N; a read with more than 20% non-ACGTN characters is dropped (reported).
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from operator import index

from ..data.inputs import normalize_file_read
from ..errors import ConfigurationError
from .sequence_ids import UniqueSequenceIds
from .sequence_records import detect_sequence_format, iter_sequence_records, open_text


def _detect_format(path: str, seq_column: str | None = None) -> str:
    if path == "-":
        return "stdin"
    with open_text(os.fspath(path)) as stream:
        head = next((line for line in stream if line.strip()), "")
    return detect_sequence_format(os.fspath(path), head, seq_column)


@dataclass(frozen=True)
class SequenceRead:
    """One accepted read, separating its output identity from its source metadata join key."""
    sequence_id: str
    sequence: str
    source_id: str | None
    source_index: int


@dataclass(frozen=True)
class SequenceReadBatch:
    """One chunk of accepted reads, and how many of the chunk's records were dropped."""
    records: tuple[SequenceRead, ...]
    n_dropped: int

    @property
    def ids(self) -> list[str]:
        """The reads' ids."""
        return [record.sequence_id for record in self.records]

    @property
    def sequences(self) -> list[str]:
        """The reads' cleaned sequences."""
        return [record.sequence for record in self.records]

    @property
    def source_ids(self) -> list[str | None]:
        """The reads' ids as the input file gave them (None when the reader assigned one)."""
        return [record.source_id for record in self.records]


def iter_sequence_batches(path: str, chunk_size: int = 20000, seq_column: str | None = None,
                          id_column: str | None = None, rejects: list | None = None) -> Iterator[SequenceReadBatch]:
    """Yield nucleotide read batches with source IDs/positions retained across all chunks.

    One allocator reserves every accepted output ID, including generated names. Rejected reads
    keep their original positions but do not reserve output IDs. Unnamed reads get seq<source_index>.
    Source IDs (possibly None/empty) remain separate from unique output IDs for metadata joins.
    """
    try:
        if isinstance(chunk_size, bool) or index(chunk_size) <= 0:
            raise ValueError
        chunk_size = index(chunk_size)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("chunk_size must be a positive integer") from exc
    records = []
    dropped = 0
    allocator = UniqueSequenceIds()
    with closing(iter_sequence_records(path, seq_column=seq_column, id_column=id_column)) as source:
        for position, (source_id, raw) in enumerate(source):
            result = normalize_file_read(raw)
            sequence, reason = result.sequence, result.error
            if sequence is None:
                dropped += 1
                if rejects is not None:
                    rejects.append({"sequence_id": source_id or f"seq{position}", "position": position,
                                    "reason": reason, "sequence": str(raw).strip()})
            else:
                output_id = allocator.assign(source_id or f"seq{position}")
                records.append(SequenceRead(output_id, sequence, source_id, position))
            if len(records) >= chunk_size:
                yield SequenceReadBatch(tuple(records), dropped)
                records, dropped = [], 0
    if records or dropped:
        yield SequenceReadBatch(tuple(records), dropped)


def iter_sequences(path: str, chunk_size: int = 20000, seq_column: str | None = None,
                   id_column: str | None = None,
                   rejects: list | None = None) -> Iterator[tuple[list[str], list[str], int]]:
    """The reads of ``path`` in chunks of ``chunk_size``: ``(ids, cleaned nucleotide sequences, n_dropped)`` per chunk.

    Use ``alignair.io.iter_sequence_batches`` when joining source metadata or retaining original source positions.
    ID state spans all chunks and grows with distinct accepted IDs; sequence payloads are chunked.
    """
    with closing(iter_sequence_batches(path, chunk_size, seq_column, id_column, rejects)) as batches:
        for batch in batches:
            yield batch.ids, batch.sequences, batch.n_dropped


def read_sequences(path: str, seq_column: str | None = None, id_column: str | None = None,
                   collect_rejects: bool = False) -> tuple[list[str], list[str], dict]:
    """Read all the reads of ``path`` at once. For large files, read chunk by chunk with ``iter_sequences``.
    Returns (ids, sequences, info) with info.n_read/n_dropped/format; when ``collect_rejects`` is set,
    info["rejects"] lists each dropped record ({id, position, reason, sequence})."""
    ids: list[str] = []
    seqs: list[str] = []
    dropped = 0
    rejects: list | None = [] if collect_rejects else None
    for cids, cseqs, drp in iter_sequences(path, chunk_size=10 ** 9, seq_column=seq_column,
                                           id_column=id_column, rejects=rejects):
        ids += cids
        seqs += cseqs
        dropped += drp
    info = {"n_read": len(ids) + dropped, "n_dropped": dropped, "format": _detect_format(path, seq_column)}
    if collect_rejects:
        info["rejects"] = rejects
    return ids, seqs, info
