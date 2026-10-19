"""Per-read observable truth: segments, junction, productivity and clone key from a simulated record.

Everything here is reachable by a perfect aligner: segments are projected onto the observed read, the
junction exists only when both conserved anchors are aligned bases of the read, and productivity is
defined only when the junction is.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from alignair_assay.cigar import format_cigar, parse_cigar, walk
from alignair_assay.reference import Reference
from alignair_assay.seqops import revcomp, translate
from alignair_assay.truth.alleles import AlleleTruth, observable_alleles
from alignair_assay.truth.boundaries import BoundaryTruth, boundary_intervals
from alignair_assay.truth.segment import ObservedSegment, check_invariants, observe_segment

NON_BIOLOGICAL_ORIENTATION = "non_biological_orientation"

# bench_v2 orientation id -> presented sequence back to the canonical (forward) frame
_TO_CANONICAL = {
    0: lambda s: s,
    1: revcomp,
    2: lambda s: revcomp(s)[::-1],  # complement only
    3: lambda s: s[::-1],  # reverse only
}


@dataclass(frozen=True)
class SegmentTruth:
    """Truth for one observed segment.

    For an inverted D (``inverted``), ``observed`` coordinates, CIGAR and germline, the boundaries and the
    allele set are all in the frame of the reverse-complemented allele, which is how the read shows it.
    """

    observed: ObservedSegment
    boundaries: BoundaryTruth
    alleles: AlleleTruth
    observed_length: int  # germline bases of the segment in the read
    inverted: bool


@dataclass(frozen=True)
class JunctionTruth:
    start: int  # read position of the V Cys codon
    end: int  # read position just after the J Trp/Phe codon
    nt: str
    aa: str | None  # None when length % 3 != 0
    length: int


@dataclass(frozen=True)
class ObservableTruth:
    case_id: str
    stratum: str
    sequence: str  # canonical (forward) read, uppercase
    orientation: int  # 0 forward, 1 reverse complement (bench ids 2/3 are excluded)
    segments: dict[str, SegmentTruth]  # only observed segments
    # Genes the simulator called whose truth could not be built (reason in ``warnings``); never scored.
    unscorable: frozenset[str]
    junction: JunctionTruth | None
    vj_in_frame: bool | None
    stop_codon: bool | None
    productive: bool | None
    clone_key: tuple[frozenset[str], frozenset[str], int] | None
    v_mismatches: int | None
    excluded: str | None
    warnings: tuple[str, ...]


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
                seg = _observe_inverted_d(read, allele_seq, primary, record, problems)
            else:
                seg = _observe_forward(read, allele_seq, primary, record, g, problems)
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
    vj_in_frame = stop_codon = productive = clone_key = None
    if junction is not None:
        vj_in_frame = junction.length % 3 == 0
        stop_codon = _stop_codon(read, junction.start, segments["v"].observed.seq_start,
                                 segments["j"].observed.seq_end)
        productive = vj_in_frame and not stop_codon
        clone_key = (segments["v"].alleles.genes, segments["j"].alleles.genes, junction.length)

    return ObservableTruth(
        case_id=case_id, stratum=stratum, sequence=read, orientation=orientation, segments=segments,
        unscorable=frozenset(unscorable), junction=junction, vj_in_frame=vj_in_frame, stop_codon=stop_codon,
        productive=productive, clone_key=clone_key,
        v_mismatches=segments["v"].observed.n_mismatch if "v" in segments else None,
        excluded=None, warnings=tuple(warnings),
    )


def truth_from_bench_case(case: dict, reference: Reference) -> ObservableTruth:
    """Truth for one bench_v2 JSONL row, in the canonical frame of its forward ``record``.

    Complement-only and reverse-only presentations (orientation ids 2/3) are built but marked excluded.
    """
    oid = case["orientation_id"]
    if oid not in _TO_CANONICAL:
        raise ValueError(f"{case['case_id']}: unknown orientation_id {oid!r}")
    truth = truth_from_genairr_record(case["record"], reference, case_id=case["case_id"],
                                      stratum=case["stratum"], orientation=oid)
    if _TO_CANONICAL[oid](case["sequence"]).upper() != truth.sequence:
        raise ValueError(f"{case['case_id']}: presented sequence does not match orientation_id {oid}")
    if oid in (2, 3):
        truth = replace(truth, excluded=NON_BIOLOGICAL_ORIENTATION)
    return truth


def _observe_forward(read: str, allele_seq: str, allele: str, record: dict, g: str,
                     problems: list[str]) -> ObservedSegment | None:
    """Observed alignment of a forward segment, or None (with a problem) if the record's alignment is
    inconsistent with the read or the allele.

    GenAIRR occasionally reports an alignment that runs past the allele end (an extra read base attributed to
    the segment after an insertion), which cannot be projected onto the reference.
    """
    seq_start, germ_start = record[f"{g}_sequence_start"], record[f"{g}_germline_start"]
    cigar = record[f"{g}_cigar"]
    ops = parse_cigar(cigar)
    read_end = seq_start + sum(n for n, op in ops if op in "MI=X")
    germ_end = germ_start + sum(n for n, op in ops if op in "MD=X")
    where = f"simulator alignment {cigar} from"
    if seq_start < 0 or germ_start < 0:
        problems.append(f"{where} read {seq_start} / germline {germ_start} starts before 0; segment dropped")
    elif germ_end > len(allele_seq):
        problems.append(f"{where} germline {germ_start} ends at {germ_end}, past the end of {allele} "
                        f"({len(allele_seq)} nt); segment dropped")
    elif read_end > len(read):
        problems.append(f"{where} read {seq_start} ends at {read_end}, past the read end ({len(read)} nt); "
                        f"segment dropped")
    else:
        return observe_segment(read, allele_seq, allele, seq_start, germ_start, cigar)
    return None


def _observe_inverted_d(read: str, allele_seq: str, allele: str, record: dict,
                        problems: list[str]) -> ObservedSegment | None:
    """Observed alignment of an inverted D against ``revcomp(allele_seq)``.

    GenAIRR reports an inverted D in forward-allele coordinates and walks its descending germline positions
    with a forward-only CIGAR walker, so: every ``D`` op is an artifact (a fill from the 5' trim up to the
    highest observed base and from the lowest observed base to the 3' trim), deletions inside the D are not
    recorded, and NP-extension claims (``M`` ops below the 5' trim / above the 3' trim) are scored against
    the forward allele. The ``M``/``I`` ops are still the read bases in order, which is enough to rebuild the
    alignment; unrecorded internal deletions are placed where they minimise mismatches.
    """
    n_allele = len(allele_seq)
    trim_5, end_germ = record["d_trim_5"], n_allele - record["d_trim_3"]
    seq_start = record["d_sequence_start"]
    cols = [op for n, op in parse_cigar(record["d_cigar"]) for _ in range(n)]
    read_cols = [k for k, op in enumerate(cols) if op != "D"]
    if len(read_cols) != record["d_sequence_end"] - seq_start:
        problems.append(f"inverted D CIGAR {record['d_cigar']} does not cover the read span; segment dropped")
        return None
    unexpected = f"inverted D CIGAR {record['d_cigar']} has an unexpected shape; segment dropped"
    # NP claims: one M per germline position below the 5' trim (head) / at or above the 3' trim (tail).
    n_head = max(trim_5 - record["d_germline_start"], 0)
    n_tail = max(record["d_germline_end"] - end_germ, 0)
    claims = read_cols[:n_head] + read_cols[len(read_cols) - n_tail:]
    if n_head + n_tail > len(read_cols) or any(cols[k] != "M" for k in claims):
        problems.append(unexpected)
        return None
    body = read_cols[n_head:len(read_cols) - n_tail]
    m_cols = [k for k in body if cols[k] == "M"]
    if not m_cols:
        return None
    first = trim_5 + cols[:m_cols[0]].count("D")  # highest forward germline position observed
    last = end_germ - 1 - cols[m_cols[-1] + 1:].count("D")  # lowest
    n_del = (first - last + 1) - len(m_cols)
    if "D" in cols[m_cols[0]:m_cols[-1]] or n_del < 0:
        problems.append(unexpected)
        return None
    seq_start += n_head
    rc_seq = revcomp(allele_seq)
    rc_start = n_allele - 1 - first
    ops = [cols[k] for k in body]
    if n_del:
        ops = _place_deletions(read, rc_seq, seq_start, rc_start, ops, n_del)
        if ops is None:
            problems.append(unexpected)
            return None
        problems.append(f"inverted D: {n_del} unrecorded internal deletion(s) placed by minimum mismatch")
    return observe_segment(read, rc_seq, allele, seq_start, rc_start, format_cigar([(1, op) for op in ops]))


def _place_deletions(read: str, germline: str, seq_start: int, germ_start: int, ops: list[str],
                     n_del: int) -> list[str] | None:
    """Insert ``n_del`` germline-only columns strictly inside ``ops`` (M/I) minimising mismatches.

    Ties go to the leftmost placement. None when there is no internal slot (fewer than two ``M``).
    """
    n = len(ops)
    m_before = [0]
    for op in ops:
        m_before.append(m_before[-1] + (op == "M"))
    last_m = max(i for i, op in enumerate(ops) if op == "M")
    inf = float("inf")
    cost = [[inf] * (n_del + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (n_del + 1) for _ in range(n + 1)]
    cost[0][0] = 0
    for i in range(n + 1):
        for d in range(n_del + 1):
            c = cost[i][d]
            if c == inf:
                continue
            if d < n_del and m_before[i] >= 1 and i <= last_m and c < cost[i][d + 1]:
                cost[i][d + 1], back[i][d + 1] = c, "D"
            if i < n:
                add = 0
                if ops[i] == "M":
                    b = read[seq_start + i]
                    add = int(b != "N" and b != germline[germ_start + m_before[i] + d])
                if c + add < cost[i + 1][d]:
                    cost[i + 1][d], back[i + 1][d] = c + add, ops[i]
    if cost[n][n_del] == inf:
        return None
    out: list[str] = []
    i, d = n, n_del
    while i or d:
        step = back[i][d]
        out.append(step)
        if step == "D":
            d -= 1
        else:
            i -= 1
    return out[::-1]


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


def _stop_codon(read: str, junction_start: int, v_start: int, j_end: int) -> bool:
    """Stop in the junction's frame over full codons from the first V base to the last J base."""
    first = v_start + (junction_start - v_start) % 3
    n_codons = max((j_end - first) // 3, 0)
    return "*" in translate(read[first:first + 3 * n_codons])
