"""``predict --infer-genotype``: the lean genotype pass (``genotype.lean``) around the streaming predict.

Pass 1, the plain streaming predict into a temporary TSV, hands each chunk's rows to an ``Observer`` (their V calls,
and the assembly outcome of every row with tied alleles). The genotype is inferred from them; pass 2 reads the input
again, re-calls only the moved reads with the genotype as the ``--genotype`` constraint (mask) and writes their rows;
the output is pass 1's rows with the moved ones replaced (``merge``, atomic), and the genotype is written as an AIRR
GenotypeSet (``write_genotype``). ``complete`` runs everything after pass 1.
"""
from __future__ import annotations

import csv
import json
import os
from contextlib import closing

from ..genotype.lean import GenotypeEvidence, allowed_v


def _outcome(record: dict) -> tuple:
    """A row's assembly status and, for a partial or failed row, the reason the run metadata counts."""
    status = record.get("airr_assembly_status")
    if status == "partial":
        return status, record.get("airr_assembly_reason", "unknown")
    if status == "failed":
        return status, record.get("airr_assembly_error", "?").split(":")[0]
    return status, None


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
                self.outcomes[start + offset] = _outcome(record)


def collect(input_path: str, rows, *, chunk_size: int, seq_column, id_column) -> tuple:
    """``(ids, sequences, source_ids)`` of the accepted reads at the row indices ``rows``, in order, read again from
    ``input_path`` (the reader accepts and names the same reads in the same order on every pass)."""
    from ..io.sequence_reader import iter_sequence_batches
    wanted, index = set(rows), 0
    ids, sequences, sources = [], [], []
    with closing(iter_sequence_batches(input_path, chunk_size=chunk_size, seq_column=seq_column,
                                       id_column=id_column)) as batches:
        for batch in batches:
            for read_id, sequence, source in zip(batch.ids, batch.sequences, batch.source_ids):
                if index in wanted:
                    ids.append(read_id), sequences.append(sequence), sources.append(source)
                index += 1
    return ids, sequences, sources


def recall(predict, reads: tuple, *, out_path: str, columns, extra_cols, out_locus: str, meta_by_id,
           chunk_size: int) -> list:
    """Predict the reads ``(ids, sequences, source_ids)`` with ``predict(sequences) -> records`` (the genotype
    already bound) and write their rows to ``out_path``; returns their assembly outcomes, in order."""
    from ..io.airr import AirrWriter
    from ..io.metadata import lookup_metadata_rows
    ids, sequences, sources = reads
    outcomes = []
    with AirrWriter(out_path, locus=out_locus, columns=columns, extra_columns=extra_cols) as writer:
        for start in range(0, len(sequences), chunk_size):
            chunk = slice(start, start + chunk_size)
            records = predict(sequences[chunk])
            writer.write(ids[chunk], sequences[chunk], records, metas=lookup_metadata_rows(meta_by_id, sources[chunk]))
            outcomes += [_outcome(record) for record in records]
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


def recount(counts: dict, fail_reasons: dict, partial_reasons: dict, before: list, after: list) -> None:
    """Move the replaced rows' assembly outcomes (``before``) to their new ones (``after``) in the run's counts."""
    for sign, outcomes in ((-1, before), (1, after)):
        for status, reason in outcomes:
            if status in ("complete", "partial", "failed"):
                counts[status] += sign
            reasons = partial_reasons if status == "partial" else fail_reasons if status == "failed" else None
            if reasons is not None:
                reasons[reason] = reasons.get(reason, 0) + sign
                if not reasons[reason]:
                    del reasons[reason]


def write_genotype(path: str, genotype: dict, locus: str) -> None:
    """The genotype ``{gene: alleles}`` as an AIRR GenotypeSet JSON at ``path``."""
    from ..genotype.airr import to_genotype_set
    from ..genotype.zygosity import GeneCall
    zygosity = {1: "homozygous", 2: "heterozygous"}
    calls = [GeneCall(gene=gene, alleles=[{"name": allele} for allele in alleles],
                      zygosity=zygosity.get(len(alleles), "multi-allelic/duplication-candidate"),
                      deletion_candidate=False, reasons=[], novel=[]) for gene, alleles in genotype.items()]
    genotype_set, _ = to_genotype_set(calls, locus, set_id="alignair-genotype-pass")
    with open(path, "w") as handle:
        json.dump(genotype_set, handle, indent=2)


def complete(observer: Observer, args, *, first: str, predict, v_names, extra_cols, out_locus: str, meta_by_id,
             counts: dict, fail_reasons: dict, partial_reasons: dict) -> dict:
    """After pass 1 wrote ``first``: infer the genotype from ``observer``, re-call the rows it moves with
    ``predict(sequences, **genotype_constraint) -> records`` into a second temporary TSV, write ``args.out`` (pass
    1's rows with those swapped in; ``first`` itself when none moves), correct the assembly counts, and write the
    genotype. ``first`` and the second TSV are removed. Returns the run metadata's record of the pass."""
    genotype = observer.evidence.genotype(explain=args.genotype_explain, min_reads=args.genotype_min_reads)
    rows = observer.evidence.moved(genotype)
    if not rows:
        os.replace(first, args.out)
    else:
        bound = {"genotype": {"V": set(allowed_v(genotype, v_names))}, "genotype_method": "mask"}
        second = f"{args.out}.pass2.{os.getpid()}"
        try:
            reads = collect(args.input, rows, chunk_size=args.chunk_size, seq_column=args.sequence_column,
                            id_column=args.id_column)
            after = recall(lambda sequences: predict(sequences, **bound), reads, out_path=second, columns=args.columns,
                           extra_cols=extra_cols, out_locus=out_locus, meta_by_id=meta_by_id,
                           chunk_size=args.chunk_size)
            merge(first, second, rows, args.out)
        finally:
            for path in (first, second):
                if os.path.exists(path):
                    os.remove(path)
        recount(counts, fail_reasons, partial_reasons, [observer.outcomes[i] for i in rows], after)
    genotype_out = args.genotype_out or f"{args.out}.genotype.json"
    write_genotype(genotype_out, genotype, out_locus)
    return {"genotype_out": genotype_out, "explain": args.genotype_explain, "min_reads": args.genotype_min_reads,
            "genotype": genotype, "moved_rows": len(rows)}
