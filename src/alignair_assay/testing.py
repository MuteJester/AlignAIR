"""Test helpers: the prediction a perfect aligner would make for a read, and an AIRR TSV writer."""
from __future__ import annotations

import csv
from collections.abc import Iterable

from alignair_assay.ingest.airr import GENES, PredSegment, Prediction
from alignair_assay.seqops import revcomp
from alignair_assay.truth.build import ObservableTruth

_GENE_FIELDS = ("call", "call_set", "sequence_start", "sequence_end", "germline_start", "germline_end", "cigar")
AIRR_COLUMNS = (
    "sequence_id", "sequence", "rev_comp", "productive", "vj_in_frame", "stop_codon",
    *(f"{g}_{f}" for g in GENES for f in _GENE_FIELDS),
    "junction", "junction_aa", "junction_start", "junction_end",
)


def oracle_prediction(truth: ObservableTruth) -> Prediction:
    """A perfect prediction: observed coordinates and CIGAR, the full truth allele set (primary first, the
    rest sorted), and the truth junction and flags. Coordinates are in the canonical frame; ``sequence`` is
    the read as presented to the tool.
    """
    segments = {}
    for g, s in truth.segments.items():
        o, primary = s.observed, s.alleles.primary
        segments[g] = PredSegment(
            call=primary,
            call_set=(primary, *sorted(s.alleles.alleles - {primary})),
            seq_start=o.seq_start, seq_end=o.seq_end, germ_start=o.germ_start, germ_end=o.germ_end,
            cigar=o.cigar,
        )
    j = truth.junction
    rev_comp = truth.orientation == 1
    return Prediction(
        sequence_id=truth.case_id,
        sequence=revcomp(truth.sequence) if rev_comp else truth.sequence,
        rev_comp=rev_comp,
        segments=segments,
        junction_nt=j.nt if j else None,
        junction_aa=j.aa if j else None,
        junction_start=j.start if j else None,
        junction_end=j.end if j else None,
        productive=truth.productive,
        vj_in_frame=truth.vj_in_frame,
        stop_codon=truth.stop_codon,
    )


def write_airr_tsv(predictions: Iterable[Prediction], path) -> None:
    """Write predictions as an AIRR rearrangement TSV (1-based closed coordinates, ``T``/``F`` booleans).

    ``{g}_call`` lists the top-1 call first, then the rest of ``call_set``; ``load_airr_tsv`` reads the file back
    to the same predictions, except that genes without a segment come back as empty ``PredSegment``s and a call
    with an empty ``call_set`` comes back with the call as its set.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(AIRR_COLUMNS)
        for p in predictions:
            row = {
                "sequence_id": p.sequence_id, "sequence": p.sequence, "rev_comp": _flag(p.rev_comp),
                "productive": _flag(p.productive), "vj_in_frame": _flag(p.vj_in_frame),
                "stop_codon": _flag(p.stop_codon), "junction": p.junction_nt, "junction_aa": p.junction_aa,
                "junction_start": _start(p.junction_start), "junction_end": p.junction_end,
            }
            for g, s in p.segments.items():
                calls = [s.call, *(a for a in s.call_set if a != s.call)] if s.call is not None else []
                row.update({
                    f"{g}_call": ",".join(calls), f"{g}_call_set": ",".join(s.call_set),
                    f"{g}_sequence_start": _start(s.seq_start), f"{g}_sequence_end": s.seq_end,
                    f"{g}_germline_start": _start(s.germ_start), f"{g}_germline_end": s.germ_end,
                    f"{g}_cigar": s.cigar,
                })
            writer.writerow(["" if row.get(c) is None else str(row[c]) for c in AIRR_COLUMNS])


def _flag(value: bool | None) -> str | None:
    return None if value is None else "T" if value else "F"


def _start(value: int | None) -> int | None:
    return None if value is None else value + 1
