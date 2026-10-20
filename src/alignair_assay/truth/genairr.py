"""Decoding GenAIRR record alignments into observed segments.

A GenAIRR record reports each segment's alignment as a read start, a germline start and a CIGAR. These functions
check that alignment against the read and the allele and project it onto the observed read; an alignment that
cannot be projected gives None and a message in ``problems``.

**Re-called segments.** GenAIRR re-calls every segment after SHM (``{g}_call``, its best match to the mutated read);
the allele it sampled, the generative allele, is ``truth_{g}_call``. When the generative allele is not in the call
list, GenAIRR projects the record's alignment onto the re-called allele (its first call), but only at the edges: the
core columns keep the generative allele's germline positions, while a D fill runs to the re-called allele's end
(past the generative allele's end when that one is shorter) and NP bases matching the re-called allele are claimed
as germline. This is a projection artefact, not an alignment of either allele. The decoders undo it: a re-called
forward segment is clipped to the generative allele's structural germline range ``[trim_5, len(allele) - trim_3)``,
and a re-called inverted D has its fills decoded with the re-called allele's length. An NP base that happens to
match the generative allele still lies in the segment's boundary interval.

That both decoders reproduce GenAIRR's own generative alignment (the IR's ``Simulation.germline_position``) is
measured, not proven. On 3000 human IGH records at Poisson SHM rate 0.15 (seed 9) every re-called segment decodes
to it: V 187/187, D 182/182 and J 8/8; with every D inverted, inverted D 190/190, V 170/170 and J 4/4; no case had
a warning or an unscorable gene.

**Known limitation.** GenAIRR's fills stop at the projected allele's length. When the re-called allele is shorter
than the generative one, a generative deletion at germline positions at or past the re-call's length is not in the
CIGAR, so it cannot be recovered. GenAIRR's SHM makes no indels, so only records with indel noise
(``polymerase_indels``) can be affected.
"""
from __future__ import annotations

from alignair_assay.core.cigar import format_cigar, parse_cigar
from alignair_assay.core.seqops import revcomp
from alignair_assay.truth.segment import ObservedSegment, observe_segment

INVERTED_RECALL_UNDETERMINED = ("inverted D: re-called by the simulator, and its alignment, projected onto the "
                                "re-called allele, does not determine the observed span; segment dropped")


def observe_forward(read: str, allele_seq: str, allele: str, record: dict, g: str,
                    problems: list[str], *, recalled: bool = False) -> ObservedSegment | None:
    """Observed alignment of a forward segment, or None (with a problem) if the record's alignment is
    inconsistent with the read or the allele.

    GenAIRR occasionally reports an alignment that runs past the allele end (an extra read base attributed to
    the segment after an insertion), which cannot be projected onto the reference. ``recalled`` marks a segment
    whose ``allele`` is not in the record's call list: its alignment is first clipped to the allele's structural
    germline range ``[{g}_trim_5, len(allele_seq) - {g}_trim_3)`` (see the module docstring).
    """
    seq_start, germ_start = record[f"{g}_sequence_start"], record[f"{g}_germline_start"]
    cigar = record[f"{g}_cigar"]
    if recalled:
        low, high = record[f"{g}_trim_5"], len(allele_seq) - record[f"{g}_trim_3"]
        clipped = _clip_to_germline(seq_start, germ_start, cigar, low, high)
        if clipped is None:
            problems.append(f"re-called segment: simulator alignment {cigar} has no column in {allele}'s germline "
                            f"[{low}, {high}); segment dropped")
            return None
        seq_start, germ_start, cigar = clipped
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


def _clip_to_germline(seq_start: int, germ_start: int, cigar: str, low: int,
                      high: int) -> tuple[int, int, str] | None:
    """The alignment's columns from the first to the last whose germline position lies in ``[low, high)``, as
    ``(read start, germline start, CIGAR)``; None when no column does."""
    columns: list[tuple[str, int, int]] = []  # (op, read position, germline position) at each column
    read_pos, germ_pos = seq_start, germ_start
    for n, op in parse_cigar(cigar):
        for _ in range(n):
            columns.append((op, read_pos, germ_pos))
            read_pos += op in "MI=X"
            germ_pos += op in "MD=X"
    inside = [k for k, (op, _, position) in enumerate(columns) if op != "I" and low <= position < high]
    if not inside:
        return None
    kept = columns[inside[0]:inside[-1] + 1]
    return kept[0][1], kept[0][2], format_cigar([(1, op) for op, _, _ in kept])


def observe_inverted_d(read: str, allele_seq: str, allele: str, record: dict, problems: list[str], *,
                       projected_length: int | None = None) -> ObservedSegment | None:
    """Observed alignment of an inverted D against ``revcomp(allele_seq)``.

    GenAIRR reports an inverted D in forward-allele coordinates and walks its descending germline positions
    with a forward-only CIGAR walker, so: every ``D`` op is an artifact (a fill from the 5' trim up to the
    highest observed base and from the lowest observed base to the 3' trim), deletions inside the D are not
    recorded, and NP-extension claims (``M`` ops below the 5' trim / above the 3' trim) are scored against
    the forward allele. The ``M``/``I`` ops are still the read bases in order, which is enough to rebuild the
    alignment; unrecorded internal deletions are placed where they minimise mismatches.

    ``projected_length`` is the length of the allele GenAIRR projected the alignment onto when that is not
    ``allele`` (a re-called D, see the module docstring): the fills run to that allele's ends, so they are decoded
    with its length while positions stay in ``allele``'s frame. A fill cut short by the re-called allele's end
    leaves the observed span undetermined, and the segment is dropped with ``INVERTED_RECALL_UNDETERMINED``.
    """
    n_allele = len(allele_seq)
    walker_length = n_allele if projected_length is None else projected_length
    trim_5, end_germ = record["d_trim_5"], walker_length - record["d_trim_3"]
    seq_start = record["d_sequence_start"]
    cols = [op for n, op in parse_cigar(record["d_cigar"]) for _ in range(n)]
    read_cols = [k for k, op in enumerate(cols) if op != "D"]
    if len(read_cols) != record["d_sequence_end"] - seq_start:
        problems.append(f"inverted D CIGAR {record['d_cigar']} does not cover the read span; segment dropped")
        return None
    unexpected = f"inverted D CIGAR {record['d_cigar']} has an unexpected shape; segment dropped"
    # NP claims: one M per germline position below the 5' trim (head) / at or above the 3' trim (tail).
    n_head = max(trim_5 - record["d_germline_start"], 0)
    tail_from = end_germ
    if projected_length is not None:
        # A re-called D's own positions may run past the re-called allele's 3' trim; its claims start above them.
        first_m = next((k for k in read_cols[n_head:] if cols[k] == "M"), None)
        if first_m is not None:
            tail_from = max(end_germ, trim_5 + cols[:first_m].count("D") + 1)
    n_tail = max(record["d_germline_end"] - tail_from, 0)
    claims = read_cols[:n_head] + read_cols[len(read_cols) - n_tail:]
    if n_head + n_tail > len(read_cols) or any(cols[k] != "M" for k in claims):
        problems.append(unexpected)
        return None
    body = read_cols[n_head:len(read_cols) - n_tail]
    m_cols = [k for k in body if cols[k] == "M"]
    if not m_cols:
        return None
    first = trim_5 + cols[:m_cols[0]].count("D")  # highest forward germline position observed
    n_trailing = cols[m_cols[-1] + 1:].count("D")
    # The walker's fills stop at the projected allele's end. A 5' fill reaching it hides the highest position; with
    # the highest position at or past the projected 3' trim and no 3' fill, the lowest position is unknown.
    if projected_length is not None and (first >= walker_length or (n_trailing == 0 and first >= end_germ)):
        problems.append(INVERTED_RECALL_UNDETERMINED)
        return None
    last = end_germ - 1 - n_trailing  # lowest
    n_del = (first - last + 1) - len(m_cols)
    if "D" in cols[m_cols[0]:m_cols[-1]] or n_del < 0:
        problems.append(unexpected)
        return None
    seq_start += n_head
    rc_seq = revcomp(allele_seq)
    rc_start = n_allele - 1 - first
    ops = [cols[k] for k in body]
    if n_del:
        ops = place_deletions(read, rc_seq, seq_start, rc_start, ops, n_del)
        if ops is None:
            problems.append(unexpected)
            return None
        problems.append(f"inverted D: {n_del} unrecorded internal deletion(s) placed by minimum mismatch")
    return observe_segment(read, rc_seq, allele, seq_start, rc_start, format_cigar([(1, op) for op in ops]))


def place_deletions(read: str, germline: str, seq_start: int, germ_start: int, ops: list[str],
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
