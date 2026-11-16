"""``predict --infer-genotype``: the genotype pass (``genotype.infer``) around the streaming predict.

Pass 1, the plain streaming predict into a temporary TSV, hands each chunk's rows to an ``Observer`` (their V calls,
and the assembly outcome of every row with tied alleles). The genotype is inferred from them; pass 2 reads the input
again, re-calls only the moved reads with the genotype as the bank (bound once) and writes their rows;
the output is pass 1's rows with the moved ones replaced (``merge``, atomic), and the genotype is written as an AIRR
GenotypeSet (``genotype.airr.write_genotype_set``). ``complete`` runs everything after pass 1. Both passes read and
write as the run's ``predict_run.RunIO`` says, and pass 2 corrects the run's ``predict_run.RunCounts``.
"""
from __future__ import annotations

import csv
import os
from contextlib import closing

from ..genotype.airr import write_genotype_set
from ..genotype.infer import GenotypeEvidence, allowed_v
from .predict_run import RunCounts, RunIO, outcome


class Observer:
    """Pass 1's per-chunk hook ``(first row index, records)``: each row's V call into ``evidence``, and each row with
    tied alleles' assembly outcome into ``outcomes`` (the rows a genotype can move)."""

    def __init__(self):
        self.evidence = GenotypeEvidence()
        self.outcomes: dict[int, tuple] = {}

    def __call__(self, start: int, records: list) -> None:
        for offset, record in enumerate(records):
            v_call = record.get("v_call") or ""
            self.evidence.observe(start + offset, v_call)
            if "," in v_call:
                self.outcomes[start + offset] = outcome(record)


def collect(io: RunIO, rows) -> tuple:
    """``(ids, sequences, source_ids)`` of the accepted reads at the row indices ``rows``, in order, read again from
    the input (the reader accepts and names the same reads in the same order on every pass)."""
    from ..io.sequence_reader import iter_sequence_batches
    wanted, index = set(rows), 0
    ids, sequences, sources = [], [], []
    with closing(iter_sequence_batches(io.input_path, chunk_size=io.chunk_size, seq_column=io.seq_column,
                                       id_column=io.id_column)) as batches:
        for batch in batches:
            for read_id, sequence, source in zip(batch.ids, batch.sequences, batch.source_ids, strict=True):
                if index in wanted:
                    ids.append(read_id), sequences.append(sequence), sources.append(source)
                index += 1
    return ids, sequences, sources


def recall(predict, reads: tuple, io: RunIO, *, out_path: str) -> list:
    """Predict the reads ``(ids, sequences, source_ids)`` with ``predict(sequences) -> records`` (the genotype
    already bound) and write their rows to ``out_path``; returns their assembly outcomes, in order."""
    from ..io.airr import AirrWriter
    from ..io.metadata import lookup_metadata_rows
    ids, sequences, sources = reads
    outcomes = []
    with AirrWriter(out_path, locus=io.out_locus, columns=io.columns, extra_columns=io.extra_cols) as writer:
        for start in range(0, len(sequences), io.chunk_size):
            chunk = slice(start, start + io.chunk_size)
            records = predict(sequences[chunk])
            writer.write(ids[chunk], sequences[chunk], records,
                         metas=lookup_metadata_rows(io.meta_by_id, sources[chunk]))
            outcomes += [outcome(record) for record in records]
    return outcomes


def merge(first: str, second: str, rows, out: str) -> None:
    """Write ``out`` (atomically): ``first``'s rows, the rows at the indices ``rows`` (sorted) taken in turn from
    ``second``, which holds exactly those rows under the same header."""
    wanted, tmp = set(rows), f"{out}.tmp.{os.getpid()}"
    with open(first, newline="") as kept, open(second, newline="") as moved, open(tmp, "w", newline="") as target:
        kept_rows, moved_rows = csv.reader(kept, delimiter="\t"), csv.reader(moved, delimiter="\t")
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(next(kept_rows))
        next(moved_rows)
        for index, row in enumerate(kept_rows):
            writer.writerow(next(moved_rows) if index in wanted else row)
    os.replace(tmp, out)


def complete(observer: Observer, io: RunIO, counts: RunCounts, *, first: str, out: str, explain: float,
             min_reads: int, genotype_out: str | None, bind, v_names) -> dict:
    """After pass 1 wrote ``first``: infer the genotype from ``observer`` (``explain``, ``min_reads``), re-call the rows
    it moves with ``bind(genotype) -> predict`` (``predict(sequences) -> records`` from that genotype's bank, bound
    once) into a second temporary TSV, write ``out`` (pass 1's rows with those swapped in; ``first`` itself when none
    moves), correct ``counts``, and write the genotype to ``genotype_out`` (default ``<out>.genotype.json``). ``first``
    and the second TSV are removed. Returns the run metadata's record of the pass."""
    genotype = observer.evidence.genotype(explain=explain, min_reads=min_reads)
    rows = observer.evidence.moved(genotype)
    if not rows:
        os.replace(first, out)
    else:
        second = f"{out}.pass2.{os.getpid()}"
        try:
            predict = bind({"V": set(allowed_v(genotype, v_names))})
            after = recall(predict, collect(io, rows), io, out_path=second)
            merge(first, second, rows, out)
        finally:
            for path in (first, second):
                if os.path.exists(path):
                    os.remove(path)
        counts.move([observer.outcomes[i] for i in rows], after)
    genotype_out = genotype_out or f"{out}.genotype.json"
    write_genotype_set(genotype_out, genotype, io.out_locus)
    return {"genotype_out": genotype_out, "explain": explain, "min_reads": min_reads,
            "genotype": genotype, "moved_rows": len(rows)}
