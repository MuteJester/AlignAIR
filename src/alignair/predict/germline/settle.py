"""Settling a V alignment's ends: the start re-placed where a deletion in the segment's first bases left it past the
germline's first base (``settle_start``), and the end moved over an indel near the V's 3' end or N bases the model
stopped at (``settle_end``), each only where the alignment then scores better."""
from __future__ import annotations

from .cigar import cigar_ops, derive_cigar, joined_cigar

_SETTLE_REACH = 6      # germline bases the start may move either way when settle_start re-places it
_SETTLE_MAX_REACH = 12  # the most it may move (a CIGAR opening with a longer gap widens the reach up to this),
#                        and the most a segment's length may differ from its window's for a search at all
_SETTLE_CHECK = 8      # read bases at the start of an alignment that, holding a gap or a mismatch, open the search
_SETTLE_MIN_LENGTH = 15  # shortest segment searched (the locator's anchor length): a shorter one has no placement
_SETTLE_PREFIX = 96    # read bases of the alignment re-derived for each start searched


def _alignment_score(read: str, germ: str, cigar: str) -> int:
    """Matches minus mismatches minus 2 per gap column of ``read`` against ``germ`` (the window the CIGAR spans).
    Read bases before the first germline-aligned one (a leading insertion) cost 1 each, as a mismatch: moving a start
    past mutated first bases gains nothing, while a start a deletion pushed past the germline's first base gains 2 per
    base by moving back (its leading bases then match). Germline bases before the first aligned read base (a leading
    deletion) cost 2 each, so a window that starts too early never gains."""
    score, r, g, leading = 0, 0, 0, True
    for n, op in cigar_ops(cigar):
        if op == "M":
            same = sum(a == b for a, b in zip(read[r:r + n], germ[g:g + n]))
            score += same - (n - same)
            r, g, leading = r + n, g + n, False
        elif op == "I":
            score -= n if leading else 2 * n
            r += n
        else:
            score -= 2 * n
            g += n
    return score


def _start_is_doubtful(read: str, germ: str, cigar: str) -> bool:
    """Whether the alignment's first ``_SETTLE_CHECK`` read bases hold a gap or two mismatches (else its start stands:
    one mismatch cannot be moved away for a gain, a leading insertion costing as much)."""
    r, g, mismatches = 0, 0, 0
    for n, op in cigar_ops(cigar):
        if op != "M":
            return True
        take = min(n, _SETTLE_CHECK - r)
        mismatches += sum(a != b for a, b in zip(read[r:r + take], germ[g:g + take]))
        if mismatches >= 2:
            return True
        r, g = r + take, g + take
        if r >= _SETTLE_CHECK:
            return False
    return False


def _pinned_prefix(ops: list[tuple[int, str]], length: int) -> tuple[int, int, list, list]:
    """``(read, germline, prefix ops, suffix ops)``: the alignment cut at its first M column at least ``length`` read
    bases in (the read and germline offsets of that column), or after its last column when there is none."""
    r = g = 0
    for k, (n, op) in enumerate(ops):
        if op == "M" and r + n > length:
            cut = max(0, length - r)
            head = [*ops[:k], *([(cut, "M")] if cut else [])]
            return r + cut, g + cut, head, [(n - cut, "M"), *ops[k + 1:]]
        if op in "MI":
            r += n
        if op in "MD":
            g += n
    return r, g, list(ops), []


def settle_start(read: str, germ: str, ref_start: int, ref_end: int, indel_count, cigar: str) -> tuple[int, str]:
    """``(germline start, CIGAR)`` for a located read segment ``read`` whose alignment to ``germ[ref_start:ref_end]``
    is ``cigar``: the start within ``_SETTLE_REACH`` bases either way (more when the CIGAR opens with a longer gap)
    whose re-derived alignment scores best (``_alignment_score``), keeping ``ref_start`` unless another scores
    strictly better (nearer starts are tried first, the earlier on a tie). A segment whose length differs from its
    window's by more than ``_SETTLE_MAX_REACH``, or shorter than ``_SETTLE_MIN_LENGTH``, is a placement no nearby
    start repairs, and is returned as it is.

    The locator derives the start from the end anchor and the segment's length, so a deletion in the segment's first
    bases leaves the start past the germline's first base by the deleted length, and the CIGAR then opens with read
    bases on no germline (``2I14M1D...``). Reads whose alignment opens clean (no gap and at most one mismatch in
    its first ``_SETTLE_CHECK`` bases) are returned as they are, without a search."""
    if (ref_end <= 0 or len(read) < _SETTLE_MIN_LENGTH or abs(len(read) - (ref_end - ref_start)) > _SETTLE_MAX_REACH
            or not _start_is_doubtful(read, germ[ref_start:ref_end], cigar)):
        return ref_start, cigar
    ops = cigar_ops(cigar)
    lead = ops[0][0] if ops and ops[0][1] != "M" else 0
    reach = min(max(_SETTLE_REACH, lead + 2), _SETTLE_MAX_REACH)
    # only the alignment's first _SETTLE_PREFIX read bases are re-derived per start: past them it is the same for
    # every start, pinned at one of its M columns
    pin_r, pin_g, head, tail = _pinned_prefix(ops, _SETTLE_PREFIX)
    prefix, pin = read[:pin_r], ref_start + pin_g
    best_score = _alignment_score(prefix, germ[ref_start:pin], joined_cigar(head, []))
    best = None
    for distance in range(1, reach + 1):
        for start in (ref_start - distance, ref_start + distance):
            if start < 0 or start >= pin:
                continue
            window = germ[start:pin]
            candidate = derive_cigar(prefix, window, indel_count)
            score = _alignment_score(prefix, window, candidate)
            if score > best_score:
                best_score, best = score, (start, candidate)
    if best is None:
        return ref_start, cigar
    return best[0], joined_cigar(cigar_ops(best[1]), tail)


_END_REACH = 12        # read and germline bases a V end may move either way
_END_GAIN = 2          # alignment score a moved end must gain over the model's
_END_ANCHOR = 8        # read bases checked at and after the end, and anchored before the reach
_END_BAND = 4          # diagonals either side of the alignment's path the end search keeps to
_END_GATE_MATCHES = 4  # of the _END_ANCHOR read bases after the end that, matching the germline after it (up to 3 bases
#                        either way), open the search


def _end_is_doubtful(read: str, germ: str, seq_end: int, ref_end: int, cigar: str) -> bool:
    """Whether a V end could move: the alignment's last ``_END_ANCHOR`` read bases hold a gap or two mismatches, or
    ``_END_GATE_MATCHES`` of the ``_END_ANCHOR`` read bases after the end match the germline after it, in place or up
    to 3 bases either way (else no search: on simulated reads such an end never gains ``_END_GAIN``)."""
    r, g, seen, mismatches = seq_end, ref_end, 0, 0
    for n, op in reversed(cigar_ops(cigar)):
        if op != "M":
            return True
        take = min(n, _END_ANCHOR - seen)
        mismatches += sum(a != b for a, b in zip(read[r - take:r], germ[g - take:g]))
        r, g, seen = r - take, g - take, seen + take
        if mismatches >= 2:
            return True
        if seen >= _END_ANCHOR:
            break
    after = read[seq_end:seq_end + _END_ANCHOR]
    for shift in range(-3, 4):
        window = germ[ref_end + shift:ref_end + shift + _END_ANCHOR]
        if ref_end + shift >= 0 and sum(a == b for a, b in zip(after, window)) >= _END_GATE_MATCHES:
            return True
    return False


def _tail_anchor(seq_start: int, seq_end: int, ref_start: int, cigar: str, back: int) -> tuple[int, int]:
    """``(read, germline)`` position of an M column of the alignment about ``back`` read bases before its end (its
    start when it is shorter)."""
    target = seq_end - back
    r, g = seq_start, ref_start
    for n, op in cigar_ops(cigar):
        if op == "M":
            if r + n > target:
                k = max(0, target - r)
                return r + k, g + k
            r, g = r + n, g + n
        elif op == "I":
            r += n
        else:
            g += n
    return r, g


def settle_end(read: str, germ: str, seq_start: int, seq_end: int, ref_start: int, ref_end: int, cigar: str,
               limit: int) -> tuple[int, int]:
    """``(seq_end, ref_end)`` of a V alignment of ``read[seq_start:seq_end]`` (the whole read, uppercase) on
    ``germ[ref_start:ref_end]`` (CIGAR ``cigar``), its end moved where the alignment scores best.

    The model's V end can stop short of an indel near the V's 3' end, or run into N bases. The alignment is anchored
    about ``_END_REACH + _END_ANCHOR`` read bases before its end and extended by a banded DP (+1 a match, -1 a
    mismatch, -2 a gap column; ``_END_BAND``, ``_end_scores``) over the read and germline bases within ``_END_REACH``
    of the end, the read no further than ``limit`` (the next segment's start); the best-scoring end (the nearest on a
    tie) replaces the model's only when it gains at least ``_END_GAIN``. Without a doubtful end (``_end_is_doubtful``)
    nothing is searched."""
    if seq_end - seq_start < _SETTLE_MIN_LENGTH or not _end_is_doubtful(read, germ, seq_end, ref_end, cigar):
        return seq_end, ref_end
    rp, gp = _tail_anchor(seq_start, seq_end, ref_start, cigar, _END_REACH + _END_ANCHOR)
    r_max, g_max = min(len(read), limit, seq_end + _END_REACH), min(len(germ), ref_end + _END_REACH)
    ci, cj = seq_end - rp, ref_end - gp
    m, n = r_max - rp, g_max - gp
    if not (0 <= ci <= m and 0 <= cj <= n):
        return seq_end, ref_end
    current, best_key, best_cell = _end_scores(read[rp:r_max], germ[gp:g_max], ci, cj)
    if current <= _UNREACHED // 2 or best_key is None or best_key[0] - current < _END_GAIN:
        return seq_end, ref_end
    return rp + best_cell[0], gp + best_cell[1]


_UNREACHED = -(1 << 30)  # the score of a cell outside the band


def _end_scores(r: str, q: str, ci: int, cj: int) -> tuple[int, tuple[int, int] | None, tuple[int, int]]:
    """The banded DP of the read bases ``r`` on the germline bases ``q`` from the anchor (their first bases): the score
    of the model's end ``(ci, cj)``, and the best end within ``_END_REACH`` of it, as ``(score, -distance)`` (the
    nearest wins a tie) and its cell. The DP keeps to the cells within ``_END_BAND`` diagonals of the anchor's and the
    end's (a V tail holds few indels)."""
    m, n = len(r), len(q)
    lo_d, hi_d = min(0, ci - cj) - _END_BAND, max(0, ci - cj) + _END_BAND
    none = _UNREACHED
    prev, current, best_key, best_cell = [none] * (n + 1), none, None, (ci, cj)
    for i in range(m + 1):
        row = [none] * (n + 1)
        for j in range(max(0, i - hi_d), min(n, i - lo_d) + 1):
            if i == 0 and j == 0:
                value = 0
            else:
                value = none
                if i and j:
                    value = prev[j - 1] + (1 if r[i - 1] == q[j - 1] else -1)
                if i and prev[j] - 2 > value:
                    value = prev[j] - 2                                   # read base i-1 on no germline base
                if j and row[j - 1] - 2 > value:
                    value = row[j - 1] - 2                                # germline base j-1 on no read base
            row[j] = value
            if i >= ci - _END_REACH and j >= cj - _END_REACH and value > none // 2:
                key = (value, -(abs(i - ci) + abs(j - cj)))
                if best_key is None or key > best_key:
                    best_key, best_cell = key, (i, j)
        if i == ci:
            current = row[cj]
        prev = row
    return current, best_key, best_cell
