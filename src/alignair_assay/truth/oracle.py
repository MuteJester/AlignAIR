"""The prediction a perfect aligner would make for a read: the reference point every metric is checked against."""
from __future__ import annotations

from alignair_assay.core.prediction import PredSegment, Prediction
from alignair_assay.core.seqops import revcomp
from alignair_assay.truth.model import ObservableTruth


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
