"""Alignment-content metrics: mutation counts, germline reconstruction and indels.

These drive SHM and selection analyses, which count mutations against the germline the tool reconstructed.
A prediction's alignment is its normalized CIGAR walked from ``seq_start``/``germ_start`` over the canonical
read and the called allele's reference sequence; read ``N`` bases are never mismatches.

Mutation and germline metrics are reachable by a perfect aligner: any segment edge inside the truth equivalence
interval is accepted. For a segment whose truth alignment has an indel they are reported under ``{g}_indel.*``
and are informational only, because indel placement is not unique.
"""
from __future__ import annotations

from alignair_assay.core.cigar import Op, parse_cigar, walk
from alignair_assay.core.prediction import PredSegment
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import CaseScores
from alignair_assay.truth.model import SegmentTruth

NAN = float("nan")


def germline_metrics(g: str, truth: SegmentTruth, read: str, pred: PredSegment,
                     reference: Reference) -> CaseScores:
    """Mutation-count and germline-reconstruction errors for one observed segment.

    Names are ``{g}.*`` when the truth alignment has no indel and ``{g}_indel.*`` (informational) otherwise.

    ``mutation_count_err`` is |predicted mismatches - truth mismatches|. Truth mismatches are counted over the
    truth columns whose read position lies between the predicted edges, each edge taken as predicted when it is
    inside its truth interval and as the truth edge otherwise: columns a prediction legitimately leaves out (the
    mismatching shrink zone) are not counted, and columns it legitimately adds (the extension zone) match.

    ``germline_errors`` counts the truth columns of the mandatory region (read positions between the inner ends of
    the start and end intervals) whose predicted germline base is missing or wrong, plus the remaining truth
    columns the prediction aligns to a wrong germline base; a wrong sibling allele or a shifted germline
    coordinate shows up.

    Both are ``nan`` when the prediction has no usable alignment (no call, a call not in the reference, missing
    coordinates or CIGAR, or a walk outside the read or the allele).
    """
    obs = truth.observed
    prefix = f"{g}_indel" if any(_indel_counts(parse_cigar(obs.cigar))) else g
    names = (f"{prefix}.mutation_count_err", f"{prefix}.germline_errors")
    aligned = predicted_germline(g, read, pred, reference)
    if aligned is None:
        return dict.fromkeys(names, NAN)
    predicted, pred_start, pred_end = aligned
    b = truth.boundaries
    start = pred_start if _inside(pred_start, b.seq_start) else obs.seq_start
    end = pred_end if _inside(pred_end, b.seq_end) else obs.seq_end
    mandatory_start, mandatory_end = b.seq_start[1], b.seq_end[0]

    pred_mismatches = sum(read[r] != "N" and read[r] != base for r, base in predicted.items())
    truth_mismatches = errors = 0
    for c in walk(parse_cigar(obs.cigar), obs.seq_start, obs.germ_start):
        if c.read_pos is None or c.germ_pos is None:
            continue
        r, germline = c.read_pos, obs.observed_germline[c.germ_pos - obs.germ_start]
        if start <= r < end and read[r] != "N" and read[r] != germline:
            truth_mismatches += 1
        got = predicted.get(r)
        if mandatory_start <= r < mandatory_end or got is not None:
            errors += got != germline
    return {names[0]: float(abs(pred_mismatches - truth_mismatches)), names[1]: float(errors)}


def indel_metrics(g: str, truth: SegmentTruth, pred: PredSegment) -> CaseScores:
    """Indel detection, scored on the same cases for every tool (applicability depends on the truth only).

    ``indel_found`` (truth alignment has an indel): 1.0 when the predicted CIGAR has exactly the truth's numbers
    of inserted and of deleted bases; a missing CIGAR is 0.0. ``indel_false`` (truth has no indel; lower is
    better): 1.0 when the predicted CIGAR reports any insertion or deletion; ``nan`` without a predicted CIGAR.
    """
    want = _indel_counts(parse_cigar(truth.observed.cigar))
    ops = _ops(pred.cigar)
    got = None if ops is None else _indel_counts(ops)
    if any(want):
        return {f"{g}.indel_found": float(got == want)}
    return {f"{g}.indel_false": NAN if got is None else float(any(got))}


def _inside(x: int, interval: tuple[int, int]) -> bool:
    return interval[0] <= x <= interval[1]


def predicted_germline(g: str, read: str, pred: PredSegment,
                       reference: Reference) -> tuple[dict[int, str], int, int] | None:
    """Read position -> germline base for every aligned column of the prediction for gene ``g`` over the canonical
    ``read``, with the predicted read span (start, end); None when the prediction has no usable alignment (no call,
    a call not in ``reference``, missing coordinates or CIGAR, or a walk outside the read or the allele)."""
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
    return out, pred.seq_start, pred.seq_start + sum(n for n, op in ops if op in "MI=X")


def _ops(cigar: str | None) -> list[Op] | None:
    if cigar is None:
        return None
    try:
        return parse_cigar(cigar)
    except ValueError:
        return None


def _indel_counts(ops: list[Op]) -> tuple[int, int]:
    """(inserted bases, deleted bases)."""
    return sum(n for n, op in ops if op == "I"), sum(n for n, op in ops if op == "D")
