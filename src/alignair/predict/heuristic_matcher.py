"""Heuristic germline matcher — a clean reimplementation of the TF ``HeuristicReferenceMatcher``.

Instead of a free global alignment (which drifts on mutated/truncated ends), this *anchors* on what
the model already knows. For each read segment it:

  1. anchors on the segment's **end** k-mer — V/J 3' ends are conserved — and slides a small offset
     window at the germline 3' end to find the best germline end position; the window spans the length
     difference plus the model's predicted **indel count** (at least ``_INDEL_REACH``);
  2. **derives the start** from the known segment length, then refines it within a window whose width
     is the predicted indel count, at least ``_INDEL_REACH``, at most the length difference.

For V, once the CIGAR is derived, :func:`settle_start` re-places a start the length derivation got wrong (a deletion
in the segment's first bases leaves it past the germline's first base) where the alignment scores best.

It is ``O(search_span · k)`` per segment — no dynamic programming — and returns germline coordinates
(no CIGAR; substitutions/indels are recovered downstream from the coordinates + the model's heads).

This is a faithful, behaviour-identical port of the TF logic, restructured for clarity: a typed
``GermlineMatch`` result, single-segment ``match_one`` split into named steps, and a ``from_reference``
constructor. See ``tests/alignair/predict/test_heuristic_matcher.py`` for the identical-results guard.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import AlignResult

_ACCEL = 0.05          # per-step velocity increment that amplifies same-state (match/mismatch) runs
_START_CHECK = 30      # segment bases compared at the placement's start when checking it for a mis-anchor
_INDEL_REACH = 3       # fewest indels the end and start searches allow for, whatever the predicted count


def _run_length_encode(ops: str) -> str:
    """'MMMIMM' -> '3M1I2M' (M = aligned column, I = read-only, D = germline-only)."""
    if not ops:
        return ""
    out, prev, count = [], ops[0], 1
    for op in ops[1:]:
        if op == prev:
            count += 1
        else:
            out.append(f"{count}{prev}"); prev, count = op, 1
    out.append(f"{count}{prev}")
    return "".join(out)


def _derive_cigar_py(read: str, germ: str, indels: int, *,
                     mismatch: int = 1, gap: int = 2, margin: int = 2) -> str:
    """Pure-Python banded global alignment of a *pinned* read/germline window -> M/I/D CIGAR string.

    An equal-length window whose mismatches cost less than an insertion-deletion pair is a single ``M``
    run (no DP): every gapped path through it costs at least that pair, so it is the DP's own optimum,
    whatever the predicted indel count. Otherwise a global alignment restricted to a narrow band
    (width = net length difference + predicted indel count + margin) places the I/D ops;
    ``mismatch``/``gap`` favour substitutions (SHM) over indels. This is the reference implementation
    and the fallback when the compiled :mod:`_derive_cy` kernel is unavailable — byte-identical output.
    """
    m, n = len(read), len(germ)
    if m == n and sum(a != b for a, b in zip(read, germ)) * mismatch < 2 * gap:
        return f"{m}M" if m else ""
    if m == 0:
        return f"{n}D" if n else ""
    if n == 0:
        return f"{m}I" if m else ""

    band = abs(m - n) + indels + margin
    lo_off, hi_off = min(0, m - n) - band, max(0, m - n) + band   # allowed range of (i - j)
    INF = float("inf")
    cost = {(0, 0): 0.0}
    back: dict = {}
    for i in range(m + 1):
        j0 = max(0, i - hi_off)
        j1 = min(n, i - lo_off)
        for j in range(j0, j1 + 1):
            if i == 0 and j == 0:
                continue
            best, op = INF, None
            if i and j:                                     # M (diagonal): match or mismatch
                c = cost.get((i - 1, j - 1), INF) + (0 if read[i - 1] == germ[j - 1] else mismatch)
                if c < best:
                    best, op = c, "M"
            if i:                                           # I (read-only base)
                c = cost.get((i - 1, j), INF) + gap
                if c < best:
                    best, op = c, "I"
            if j:                                           # D (germline-only base)
                c = cost.get((i, j - 1), INF) + gap
                if c < best:
                    best, op = c, "D"
            if op is not None:
                cost[(i, j)] = best; back[(i, j)] = op

    ops, i, j = [], m, n
    while (i, j) != (0, 0):
        op = back.get((i, j))
        if op is None:                                      # off-band fallback (shouldn't happen)
            break
        ops.append(op)
        if op == "M":
            i, j = i - 1, j - 1
        elif op == "I":
            i -= 1
        else:
            j -= 1
    return _run_length_encode("".join(reversed(ops)))


try:                                                        # optional C-speed kernel (identical output)
    from ._derive_cy import derive_cigar as _cy_derive_cigar

    def _derive_cigar(read: str, germ: str, indels: int) -> str:
        return _cy_derive_cigar(read.encode("ascii"), germ.encode("ascii"), indels)

    DERIVE_BACKEND = "cython"
except (ImportError, OSError):                              # pragma: no cover - optional compiled kernel
    _derive_cigar = _derive_cigar_py
    DERIVE_BACKEND = "python"


def derive_alignment(read_window: str, germline_window: str, indel_count=0) -> AlignResult:
    """Reconstruct a base-level alignment of a read segment against its (already located) germline
    window -> an ``AlignResult`` (M/I/D CIGAR + coords).

    The window is *pinned* by :class:`HeuristicGermlineMatcher` (both ends fixed), so it can't drift
    the way an ends-free aligner does. Uses the compiled :mod:`_derive_cy` kernel when present, else
    the identical pure-Python fallback (:data:`DERIVE_BACKEND` reports which)."""
    m, n = len(read_window), len(germline_window)
    indels = max(0, int(round(_scalar(indel_count))))
    return AlignResult(0.0, _derive_cigar(read_window, germline_window, indels), 0, m, 0, n)


_SETTLE_REACH = 6      # germline bases the start may move either way when settle_start re-places it
_SETTLE_MAX_REACH = 12  # the most it may move (a CIGAR opening with a longer gap widens the reach up to this),
#                        and the most a segment's length may differ from its window's for a search at all
_SETTLE_CHECK = 8      # read bases at the start of an alignment that, holding a gap or a mismatch, open the search
_SETTLE_MIN_LENGTH = 15  # shortest segment searched (the matcher's anchor length): a shorter one has no placement
_SETTLE_PREFIX = 96    # read bases of the alignment re-derived for each start searched


def _cigar_ops(cigar: str) -> list[tuple[int, str]]:
    import re
    return [(int(n), op) for n, op in re.findall(r"(\d+)([MID])", cigar)]


def _alignment_score(read: str, germ: str, cigar: str) -> int:
    """Matches minus mismatches minus 2 per gap column of ``read`` against ``germ`` (the window the CIGAR spans).
    Read bases before the first germline-aligned one (a leading insertion) cost 1 each, as a mismatch: moving a start
    past mutated first bases gains nothing, while a start a deletion pushed past the germline's first base gains 2 per
    base by moving back (its leading bases then match). Germline bases before the first aligned read base (a leading
    deletion) cost 2 each, so a window that starts too early never gains."""
    score, r, g, leading = 0, 0, 0, True
    for n, op in _cigar_ops(cigar):
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
    for n, op in _cigar_ops(cigar):
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


def _joined(head: list, tail: list) -> str:
    """The CIGAR of ``head`` then ``tail`` ops, equal neighbouring ops merged."""
    out: list[list] = []
    for n, op in (*head, *tail):
        if n <= 0:
            continue
        if out and out[-1][1] == op:
            out[-1][0] += n
        else:
            out.append([n, op])
    return "".join(f"{n}{op}" for n, op in out)


def settle_start(read: str, germ: str, ref_start: int, ref_end: int, indel_count, cigar: str) -> tuple[int, str]:
    """``(germline start, CIGAR)`` for a located read segment ``read`` whose alignment to ``germ[ref_start:ref_end]``
    is ``cigar``: the start within ``_SETTLE_REACH`` bases either way (more when the CIGAR opens with a longer gap)
    whose re-derived alignment scores best (``_alignment_score``), keeping ``ref_start`` unless another scores
    strictly better (nearer starts are tried first, the earlier on a tie). A segment whose length differs from its
    window's by more than ``_SETTLE_MAX_REACH``, or shorter than ``_SETTLE_MIN_LENGTH``, is a placement no nearby
    start repairs, and is returned as it is.

    The matcher derives the start from the end anchor and the segment's length, so a deletion in the segment's first
    bases leaves the start past the germline's first base by the deleted length, and the CIGAR then opens with read
    bases on no germline (``2I14M1D...``). Reads whose alignment opens clean (no gap and at most one mismatch in
    its first ``_SETTLE_CHECK`` bases) are returned as they are, without a search."""
    if (ref_end <= 0 or len(read) < _SETTLE_MIN_LENGTH or abs(len(read) - (ref_end - ref_start)) > _SETTLE_MAX_REACH
            or not _start_is_doubtful(read, germ[ref_start:ref_end], cigar)):
        return ref_start, cigar
    ops = _cigar_ops(cigar)
    lead = ops[0][0] if ops and ops[0][1] != "M" else 0
    reach = min(max(_SETTLE_REACH, lead + 2), _SETTLE_MAX_REACH)
    # only the alignment's first _SETTLE_PREFIX read bases are re-derived per start: past them it is the same for
    # every start, pinned at one of its M columns
    pin_r, pin_g, head, tail = _pinned_prefix(ops, _SETTLE_PREFIX)
    prefix, pin = read[:pin_r], ref_start + pin_g
    best_score = _alignment_score(prefix, germ[ref_start:pin], _joined(head, []))
    best = None
    for distance in range(1, reach + 1):
        for start in (ref_start - distance, ref_start + distance):
            if start < 0 or start >= pin:
                continue
            window = germ[start:pin]
            candidate = derive_alignment(prefix, window, indel_count).cigar
            score = _alignment_score(prefix, window, candidate)
            if score > best_score:
                best_score, best = score, (start, candidate)
    if best is None:
        return ref_start, cigar
    return best[0], _joined(_cigar_ops(best[1]), tail)


@dataclass(frozen=True)
class GermlineMatch:
    """Where a read segment sits in its germline. Read-frame ``seq_*`` (possibly overhang-adjusted)
    and germline-frame ``ref_*`` coordinates; all 0-based, end-exclusive."""
    seq_start: int
    seq_end: int
    ref_start: int
    ref_end: int


class HeuristicGermlineMatcher:
    def __init__(self, germlines: dict[str, str], *, anchor_k: int = 15,
                 search_span: int | None = 50, end_offset_penalty: float = 0.1,
                 fallback_rate: float = 0.3):
        """``germlines``: {allele_name -> uppercase ungapped germline}. ``anchor_k``: end-anchor k-mer
        length. ``search_span``: max offset scanned inward from the germline 3' end (default 50 — deep
        enough to find a 3'-truncated end, e.g. a read ending inside J, while bounding cost on long V;
        ``None`` = adaptive/unbounded). ``end_offset_penalty``: per-offset bias toward the germline 3'
        end — keeps 5'-truncated reads (V) anchored at their intact 3' end while still letting genuinely
        3'-truncated reads (J) move deep when the deeper match is clearly better. The TF version was
        ``search_span=30, end_offset_penalty=0``, which was blind to J 3'-truncation.

        ``fallback_rate``: the end-anchor assumes the segment's 3' end sits near the germline 3' end,
        which breaks on 3'-truncated *amplicons* (e.g. an FR1 read cut before V's 3' end) — it then
        mislocates the germline coordinates by ~the truncation length. So after anchoring we *verify*
        the placement's substitution rate and, if it exceeds ``fallback_rate`` (a clear mis-anchor, vs
        SHM's <~0.2), relocate with a whole-germline offset scan that is agnostic to which end is
        truncated. Set to 0 to disable the fallback (pure end-anchor). The same rate on the segment's tail
        tells overhang from insertions in a segment longer than its germline."""
        self._germlines = germlines
        self._anchor_k = anchor_k
        self._search_span = search_span
        self._end_offset_penalty = end_offset_penalty
        self._fallback_rate = fallback_rate

    # ------------------------------------------------------------------ scoring
    @staticmethod
    def _anchor_cost(query_kmer: str, germline_window: str) -> float:
        """Directional match cost of aligning ``query_kmer`` to ``germline_window`` position-by-position:
        matches reward, mismatches penalize, and runs of the same outcome are amplified by a growing
        velocity term. Lower is a better anchor. (Identical to the TF ``_affine_alignment_cost``.)"""
        score = velocity = 0.0
        prev_matched = None
        for matched in (q == g for q, g in zip(query_kmer, germline_window)):
            if matched:
                velocity = velocity + _ACCEL if prev_matched else _ACCEL
                score -= 1.0 + velocity
            else:
                velocity = velocity + _ACCEL if prev_matched is False else _ACCEL
                score += 1.0 + velocity
            prev_matched = matched
        return score

    @staticmethod
    def _ends_are_clean(segment: str, germline: str, k: int = 10, max_mismatch: int = 3) -> bool:
        """True iff the first and last ``k`` bases each match the germline within ``max_mismatch``."""
        head = sum(a != b for a, b in zip(segment[:k], germline[:k]))
        tail = sum(a != b for a, b in zip(segment[-k:], germline[-k:]))
        return head <= max_mismatch and tail <= max_mismatch

    @staticmethod
    def _trim_symmetric_overhang(seq_start: int, seq_end: int, germline_len: int) -> tuple[int, int]:
        """Trim an over-long, indel-free segment down to the germline length, equally from both ends."""
        excess = (seq_end - seq_start) - germline_len
        if excess <= 0:
            return seq_start, seq_end
        left = excess // 2
        return seq_start + left, seq_end - (excess - left)

    # ------------------------------------------------------------------ core search
    def _locate_in_germline(self, segment: str, germline: str, indel_count) -> tuple[int, int]:
        seg_len, ref_len = len(segment), len(germline)
        # a segment shorter than the anchor (most D segments) is its own anchor: a longer window would compare it with
        # the window's first bases only and put its end past its last base
        k = max(1, min(self._anchor_k, seg_len))
        len_diff = abs(ref_len - seg_len)
        # the indel head undercounts (reads with three indels can predict ~0.03), so both searches below reach at
        # least _INDEL_REACH bases past what the length alone allows
        indels = max(0, int(round(float(_scalar(indel_count)))))

        # 1) anchor on the END k-mer; slide the germline 3' end over the offset window. The window
        #    spans the length difference plus the indel allowance (insertions make a read longer than the
        #    germline stretch it covers, so its 3' end can sit deeper than the length difference), the
        #    deeper offsets only while the window stays inside the germline; optionally capped by
        #    ``search_span`` — the cap is what blinded the TF version to J 3'-truncation.
        span = min(len_diff + max(indels, _INDEL_REACH), max(len_diff, ref_len - k))
        if self._search_span is not None:
            span = min(span, self._search_span)
        end_kmer = segment[-k:]
        best_end, best_cost = ref_len, float("inf")
        for offset in range(span + 1):
            window = germline[ref_len - (k + offset): ref_len - offset]
            cost = self._anchor_cost(end_kmer, window) + self._end_offset_penalty * offset
            if cost < best_cost:
                best_cost, best_end = cost, ref_len - offset
                if best_cost == 0:
                    break

        # 2) derive the START from the segment length, refine within an indel-bounded window
        start_kmer = segment[:k]
        anchored_start = max(0, best_end - seg_len)
        best_start = anchored_start
        best_cost = self._anchor_cost(start_kmer, germline[anchored_start:anchored_start + k])
        # the offset penalty keeps the derived start unless a nearer k-mer match is clearly better
        reach = min(max(indels, _INDEL_REACH), len_diff)
        offsets = range(-reach - 1, reach + 1)
        for offset in offsets:
            cand = max(0, anchored_start + offset)
            window = germline[cand:cand + k]
            if len(window) != len(start_kmer):
                continue
            cost = self._anchor_cost(start_kmer, window) + abs(offset)
            if cost < best_cost:
                best_cost, best_start = cost, cand
                if best_cost == 0:
                    break
        return best_start, best_end

    def _scan_locate(self, segment: str, germline: str) -> tuple[int, int]:
        """Whole-germline offset scan: place the segment at the germline offset with the fewest
        substitutions — agnostic to which end (5' or 3') is truncated, unlike the end-anchor. Pure
        numpy, vectorized sliding window; germlines are short (<=~320bp) so this stays cheap and it
        only runs on the reads the anchor mis-placed."""
        m, n = len(segment), len(germline)
        if m == 0 or n == 0 or m >= n:
            return 0, min(m, n)
        seg = np.frombuffer(segment.encode("ascii"), dtype=np.uint8)
        ger = np.frombuffer(germline.encode("ascii"), dtype=np.uint8)
        windows = np.lib.stride_tricks.sliding_window_view(ger, m)     # (n-m+1, m)
        offset = int((windows != seg).sum(axis=1).argmin())
        return offset, offset + m

    @staticmethod
    def _start_mismatch_rate(segment: str, germline: str, ref_start: int) -> float:
        """Substitution rate of the segment's first ``_START_CHECK`` bases against the germline from the
        chosen start — a sanity check on the START (a correct one mismatches only at SHM rate, a
        mis-anchor ~randomly). Only the head is compared: past an indel every base of an ungapped
        comparison is shifted, so a whole-segment rate calls a correctly placed read with a body indel
        mis-anchored."""
        window = germline[ref_start:ref_start + _START_CHECK]
        m = min(len(segment), len(window))
        if m == 0:
            return 1.0
        return sum(a != b for a, b in zip(segment, window)) / m

    def _tail_mismatch_rate(self, segment: str, germline: str, ref_end: int) -> float:
        """Substitution rate of the segment's last ``anchor_k`` bases against the germline ending at ``ref_end``: SHM
        rate where the tail is germline, ~random where it is overhang (0 when there is no tail to compare)."""
        k = min(self._anchor_k, len(segment), ref_end)
        if k == 0:
            return 0.0
        return sum(a != b for a, b in zip(segment[-k:], germline[ref_end - k:ref_end])) / k

    # ------------------------------------------------------------------ public API
    def match_one(self, sequence: str, seq_start: int, seq_end: int, allele: str,
                  indel_count) -> GermlineMatch:
        germline = self._germlines[allele]
        ref_len = len(germline)

        # over-long indel-free segment -> symmetric trim before matching
        if indel_count == 0 and (seq_end - seq_start) > ref_len:
            seq_start, seq_end = self._trim_symmetric_overhang(seq_start, seq_end, ref_len)

        segment = sequence[seq_start:seq_end]
        # confident quick-exit: exact-length, indel-free, clean ends -> full-germline span
        if len(segment) == ref_len and indel_count == 0 and self._ends_are_clean(segment, germline):
            return GermlineMatch(seq_start, seq_end, 0, ref_len)

        orig_len = seq_end - seq_start
        ref_start, ref_end = self._locate_in_germline(segment, germline, indel_count)

        # verify the anchor's start; a 3'-truncated amplicon breaks the end-anchor's START
        # derivation (start = end - seg_len over-/under-shoots when the read overhangs the germline),
        # so if the segment's head mismatches the germline there far above SHM rate, relocate the START
        # with the truncation-agnostic whole-germline scan while KEEPING the anchor's end (which stays
        # accurate).
        if self._fallback_rate and (
                self._start_mismatch_rate(segment, germline, ref_start) > self._fallback_rate):
            scan_start, _ = self._scan_locate(segment, germline)
            if scan_start < ref_end:
                ref_start = scan_start

        # a segment longer than its germline holds insertions or overhang (read bases past the germline, e.g. a
        # constant region after J). Overhang leaves the segment's tail off the germline where the end anchor put
        # it: then the germline was trimmed only because the read stuck out -> fold that trim back into the read
        # coordinates and report a full-germline span. A tail that follows the germline is insertions, and the read
        # coordinates stay.
        pure_overhang = orig_len > ref_len or (orig_len == ref_len and indel_count == 0)
        if pure_overhang and (ref_start > 0 or ref_end < ref_len) and (
                self._tail_mismatch_rate(segment, germline, ref_end) > self._fallback_rate):
            seq_start = max(0, seq_start + ref_start)
            seq_end = min(len(sequence), seq_end - (ref_len - ref_end))
            ref_start, ref_end = 0, ref_len
        return GermlineMatch(seq_start, seq_end, ref_start, ref_end)


def _scalar(x) -> float:
    """Coerce a possibly-numpy / 0-d indel count to a Python float (matches the TF squeeze/round)."""
    try:
        import numpy as np
        return float(np.asarray(x).squeeze())
    except (ImportError, TypeError, ValueError):
        return float(x)


_END_REACH = 12        # read and germline bases a V end may move either way
_END_GAIN = 2          # alignment score a moved end must gain over the model's
_END_ANCHOR = 8        # read bases checked at and after the end, and anchored before the reach
_END_BAND = 4          # diagonals either side of the alignment's path the end search keeps to
_END_GATE_MATCHES = 4  # of the _END_ANCHOR read bases after the end that, matching the germline after it (up to 3 bases
#                        either way), open the search


def _end_is_doubtful(read: str, germ: str, seq_end: int, ref_end: int, cigar: str) -> bool:
    """Whether a V end could move: the alignment's last ``_END_ANCHOR`` read bases hold a gap or two mismatches, or
    ``_END_GATE_MATCHES`` of the ``_END_ANCHOR`` read bases after the end match the germline after it, in place or up
    to 3 bases either way (else no search: its alignment cannot gain ``_END_GAIN``, as measured on the dev suites)."""
    r, g, seen, mismatches = seq_end, ref_end, 0, 0
    for n, op in reversed(_cigar_ops(cigar)):
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
        if ref_end + shift >= 0 and sum(
                a == b for a, b in zip(after, germ[ref_end + shift:ref_end + shift + _END_ANCHOR])) >= _END_GATE_MATCHES:
            return True
    return False


def _tail_anchor(seq_start: int, seq_end: int, ref_start: int, cigar: str, back: int) -> tuple[int, int]:
    """``(read, germline)`` position of an M column of the alignment about ``back`` read bases before its end (its
    start when it is shorter)."""
    target = seq_end - back
    r, g = seq_start, ref_start
    for n, op in _cigar_ops(cigar):
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
    mismatch, -2 a gap column; ``_END_BAND``) over the read and germline bases within ``_END_REACH`` of the end, the read no further than ``limit``
    (the next segment's start); the best-scoring end (the nearest on a tie) replaces the model's only when it gains at
    least ``_END_GAIN``. Without a doubtful end (``_end_is_doubtful``) nothing is searched."""
    if seq_end - seq_start < _SETTLE_MIN_LENGTH or not _end_is_doubtful(read, germ, seq_end, ref_end, cigar):
        return seq_end, ref_end
    rp, gp = _tail_anchor(seq_start, seq_end, ref_start, cigar, _END_REACH + _END_ANCHOR)
    r_max, g_max = min(len(read), limit, seq_end + _END_REACH), min(len(germ), ref_end + _END_REACH)
    ci, cj = seq_end - rp, ref_end - gp
    m, n = r_max - rp, g_max - gp
    if not (0 <= ci <= m and 0 <= cj <= n):
        return seq_end, ref_end
    r, q = read[rp:r_max], germ[gp:g_max]
    # the DP keeps to the cells within _END_BAND diagonals of the anchor's and the end's (a V tail holds few indels)
    lo_d, hi_d = min(0, ci - cj) - _END_BAND, max(0, ci - cj) + _END_BAND
    none = -(1 << 30)
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
    if current <= none // 2 or best_key is None or best_key[0] - current < _END_GAIN:
        return seq_end, ref_end
    return rp + best_cell[0], gp + best_cell[1]
