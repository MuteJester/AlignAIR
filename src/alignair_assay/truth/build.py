"""Per-read observable truth: segments, junction, productivity and clone key from a simulated record.

Everything here is reachable by a perfect aligner: segments are projected onto the observed read, the
junction exists only when both conserved anchors are aligned bases of the read, and productivity is
defined only when the junction is.
"""
from __future__ import annotations

from alignair_assay.core.cigar import parse_cigar, walk
from alignair_assay.core.reference import Reference
from alignair_assay.core.seqops import revcomp, translate
from alignair_assay.truth.alleles import observable_alleles
from alignair_assay.truth.boundaries import boundary_intervals
from alignair_assay.truth.genairr import observe_forward, observe_inverted_d
from alignair_assay.truth.model import JunctionTruth, ObservableTruth, SegmentTruth
from alignair_assay.truth.segment import ObservedSegment, check_invariants


def truth_from_genairr_record(record: dict, reference: Reference, *, case_id: str, stratum: str,
                              orientation: int = 0) -> ObservableTruth:
    """Observable truth for a forward GenAIRR record (``rev_comp`` records are not supported yet)."""
    if record.get("rev_comp"):
        raise ValueError(f"{case_id}: reverse-complemented GenAIRR records are not supported")
    read = record["sequence"].upper()
    warnings: list[str] = []

    observed: dict[str, tuple[ObservedSegment, str, list[str], bool]] = {}
    unscorable: set[str] = set()
    for g in "vdj":
        calls = [c for c in (record.get(f"{g}_call") or "").split(",") if c]
        if not calls:
            continue
        primary = calls[0]
        if not reference.has(primary):
            warnings.append(f"{g}: primary allele {primary} not in reference; segment dropped")
            unscorable.add(g)
            continue
        allele_seq = reference.seq(primary)
        inverted = g == "d" and bool(record.get("d_inverted"))
        frame_seq = revcomp(allele_seq) if inverted else allele_seq
        problems: list[str] = []
        try:
            if inverted:
                seg = observe_inverted_d(read, allele_seq, primary, record, problems)
            else:
                seg = observe_forward(read, allele_seq, primary, record, g, problems)
        except (IndexError, TypeError, ValueError) as exc:
            seg = None
            problems.append(f"unusable simulator alignment ({exc}); segment dropped")
        # A missing segment with a problem was dropped; without one, the read simply does not show it.
        if seg is None and problems:
            unscorable.add(g)
        if seg is not None:
            problems += check_invariants(read, frame_seq, seg)
            observed[g] = (seg, frame_seq, calls, inverted)
        warnings.extend(f"{g}: {p}" for p in problems)

    order = list(observed)
    segments: dict[str, SegmentTruth] = {}
    d_reference = None
    for k, g in enumerate(order):
        seg, frame_seq, calls, inverted = observed[g]
        left = observed[order[k - 1]][0].seq_end if k else 0
        right = observed[order[k + 1]][0].seq_start if k + 1 < len(order) else len(read)
        if left > seg.seq_start:
            warnings.append(f"{g}: overlaps the preceding segment ({seg.seq_start} < {left})")
        if inverted:
            d_reference = d_reference or _reverse_complemented(reference, "d")
        segments[g] = SegmentTruth(
            observed=seg,
            boundaries=boundary_intervals(read, frame_seq, seg, left, right),
            alleles=observable_alleles(seg, d_reference if inverted else reference, g, calls),
            observed_length=seg.germ_end - seg.germ_start,
            inverted=inverted,
        )

    junction = _junction(read, segments, reference, warnings)
    v_frameshift, j_frameshift = (_frameshift(segments[g].observed) if g in segments else None for g in "vj")
    vj_in_frame = stop_codon = productive = clone_key = None
    if junction is not None:
        vj_in_frame = junction.length % 3 == 0
        stop_codon = _stop_codon(read, junction.start, segments["v"].observed.seq_start,
                                 segments["j"].observed.seq_end)
        productive = vj_in_frame and not stop_codon and not (v_frameshift or j_frameshift)
        clone_key = (segments["v"].alleles.genes, segments["j"].alleles.genes, junction.length)

    return ObservableTruth(
        case_id=case_id, stratum=stratum, sequence=read, orientation=orientation, segments=segments,
        unscorable=frozenset(unscorable), junction=junction, vj_in_frame=vj_in_frame, stop_codon=stop_codon,
        productive=productive, clone_key=clone_key,
        v_mismatches=segments["v"].observed.n_mismatch if "v" in segments else None,
        v_frameshift=v_frameshift, j_frameshift=j_frameshift, excluded=None, warnings=tuple(warnings),
    )


def _reverse_complemented(reference: Reference, letter: str) -> Reference:
    return Reference(alleles={letter: {n: revcomp(s) for n, s in reference.alleles[letter].items()}},
                     anchors={})


def _anchor_read_pos(seg: ObservedSegment, germ_pos: int) -> int | None:
    """Read position aligned to ``germ_pos`` through an ``M`` column, else None."""
    for col in walk(parse_cigar(seg.cigar), seg.seq_start, seg.germ_start):
        if col.germ_pos == germ_pos:
            return col.read_pos
    return None


def _junction(read: str, segments: dict[str, SegmentTruth], reference: Reference,
              warnings: list[str]) -> JunctionTruth | None:
    if "v" not in segments or "j" not in segments:
        return None
    v, j = segments["v"].observed, segments["j"].observed
    a_v, a_j = reference.anchors.get(v.allele), reference.anchors.get(j.allele)
    if a_v is None or a_j is None:
        warnings.append(f"no anchor for {v.allele if a_v is None else j.allele}; junction undefined")
        return None
    start = _anchor_read_pos(v, a_v)
    last = _anchor_read_pos(j, a_j + 2)
    if start is None or last is None:
        return None
    end = last + 1
    if end <= start:
        warnings.append(f"J anchor ({last}) precedes V anchor ({start}); junction undefined")
        return None
    nt = read[start:end]
    length = end - start
    return JunctionTruth(start=start, end=end, nt=nt, aa=translate(nt) if length % 3 == 0 else None,
                         length=length)


def _frameshift(seg: ObservedSegment) -> bool:
    ops = parse_cigar(seg.cigar)
    return (sum(n for n, op in ops if op == "I") - sum(n for n, op in ops if op == "D")) % 3 != 0


def _stop_codon(read: str, junction_start: int, v_start: int, j_end: int) -> bool:
    """Stop in the junction's frame over full codons from the first V base to the last J base."""
    first = v_start + (junction_start - v_start) % 3
    n_codons = max((j_end - first) // 3, 0)
    return "*" in translate(read[first:first + 3 * n_codons])
