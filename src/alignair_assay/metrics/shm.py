"""Alignment-content metrics: mutation counts, germline reconstruction and indels.

These drive SHM and selection analyses, which count mutations against the germline the tool reconstructed.
A prediction's alignment is its normalized CIGAR walked from ``seq_start``/``germ_start`` over the canonical
read and the called allele's reference sequence; read ``N`` bases are never mismatches.
"""
from __future__ import annotations

from alignair_assay.cigar import Op, parse_cigar, walk
from alignair_assay.ingest.airr import PredSegment
from alignair_assay.metrics import CaseScores
from alignair_assay.reference import Reference
from alignair_assay.truth.build import SegmentTruth

NAN = float("nan")


def germline_metrics(g: str, truth: SegmentTruth, read: str, pred: PredSegment,
                     reference: Reference) -> CaseScores:
    """Mutation-count and germline-reconstruction errors for one observed segment.

    ``mutation_count_err`` is |predicted mismatches - truth mismatches|. ``germline_errors`` counts the
    truth-aligned read positions whose predicted germline base is wrong or missing (the position is not
    aligned by the prediction), so a wrong sibling allele or a shifted coordinate shows up. Both are ``nan``
    when the prediction has no usable alignment (no call, a call not in the reference, missing coordinates
    or CIGAR, or a walk outside the read or the allele).
    """
    predicted = _predicted_germline(g, read, pred, reference)
    if predicted is None:
        return {f"{g}.mutation_count_err": NAN, f"{g}.germline_errors": NAN}
    mismatches = sum(read[r] != "N" and read[r] != base for r, base in predicted.items())
    obs = truth.observed
    errors = sum(predicted.get(c.read_pos) != obs.observed_germline[c.germ_pos - obs.germ_start]
                 for c in walk(parse_cigar(obs.cigar), obs.seq_start, obs.germ_start)
                 if c.read_pos is not None and c.germ_pos is not None)
    return {f"{g}.mutation_count_err": float(abs(mismatches - obs.n_mismatch)),
            f"{g}.germline_errors": float(errors)}


def indel_metric(g: str, truth: SegmentTruth, pred: PredSegment) -> CaseScores:
    """1.0 when the predicted CIGAR has as many inserted plus deleted bases as the observed truth CIGAR.

    Scored only when either side has an indel; a prediction without a CIGAR then counts as wrong.
    """
    want = _indel_bases(parse_cigar(truth.observed.cigar))
    ops = _ops(pred.cigar)
    got = None if ops is None else _indel_bases(ops)
    if not want and not got:
        return {}
    return {f"{g}.indel_exact": float(got == want)}


def _predicted_germline(g: str, read: str, pred: PredSegment, reference: Reference) -> dict[int, str] | None:
    """Read position -> germline base for every aligned column of the prediction, or None if unusable."""
    germline = reference.alleles.get(g, {}).get(pred.call) if pred.call is not None else None
    ops = _ops(pred.cigar)
    if germline is None or ops is None or pred.seq_start is None or pred.germ_start is None:
        return None
    try:
        columns = walk(ops, pred.seq_start, pred.germ_start)
    except ValueError:
        return None
    out: dict[int, str] = {}
    for c in columns:
        if c.read_pos is None or c.germ_pos is None:
            continue
        if not (0 <= c.read_pos < len(read) and 0 <= c.germ_pos < len(germline)):
            return None
        out[c.read_pos] = germline[c.germ_pos]
    return out


def _ops(cigar: str | None) -> list[Op] | None:
    if cigar is None:
        return None
    try:
        return parse_cigar(cigar)
    except ValueError:
        return None


def _indel_bases(ops: list[Op]) -> int:
    return sum(n for n, op in ops if op in "ID")
