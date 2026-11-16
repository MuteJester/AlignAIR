"""Locating a read segment in its called allele's germline, anchored on what the model already gives.

For a segment it anchors the germline end on the segment's end k-mer (V and J 3' ends are conserved), sliding an offset
window at the germline 3' end whose span is the length difference plus the model's predicted indel count (at least
``_INDEL_REACH``); it derives the start from the segment length and refines it within the indel count. A start whose
first bases mismatch far above SHM is relocated by a scan of the whole germline, and a segment longer than its germline
is told overhang from insertions by its tail. It is O(search span * k) per segment, no dynamic programming; the CIGAR is
derived on the located window (``cigar.derive_cigar``) and its ends settled after (``settle``)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cigar import indel_allowance

_CLEAN_END = 10         # bases compared at each end of an equal-length placement
_CLEAN_END_MISMATCHES = 3   # the most mismatches either end may hold and still be clean
_ACCEL = 0.05          # per-step velocity increment that amplifies same-state (match/mismatch) runs
_START_CHECK = 30      # segment bases compared at the placement's start when checking it for a mis-anchor
_INDEL_REACH = 3       # fewest indels the end and start searches allow for, whatever the predicted count


@dataclass(frozen=True)
class GermlineMatch:
    """Where a read segment sits in its germline. Read-frame ``seq_*`` (possibly overhang-adjusted)
    and germline-frame ``ref_*`` coordinates; all 0-based, end-exclusive."""
    seq_start: int
    seq_end: int
    ref_start: int
    ref_end: int


# The locator's settings. _ANCHOR_K: the end-anchor k-mer length. _SEARCH_SPAN: the deepest offset scanned inward from
# the germline 3' end (deep enough to find a 3'-truncated end, e.g. a read ending inside J, while bounding the cost on
# a long V). _END_OFFSET_PENALTY: a per-offset bias toward the germline 3' end, which keeps 5'-truncated reads (V)
# anchored at their intact 3' end while letting genuinely 3'-truncated reads (J) move deep when the deeper match is
# clearly better. _FALLBACK_RATE: the end-anchor assumes the segment's 3' end sits near the germline 3' end, which
# breaks on 3'-truncated amplicons (an FR1 read cut before V's 3' end) and then misplaces the germline coordinates by
# about the truncation; a placement whose substitution rate exceeds this (a clear mis-anchor; SHM stays well below)
# is relocated with a whole-germline offset scan that does not care which end is truncated. The same rate on the
# segment's tail tells overhang from insertions in a segment longer than its germline.
_ANCHOR_K = 15
_SEARCH_SPAN = 50
_END_OFFSET_PENALTY = 0.1
_FALLBACK_RATE = 0.3


class GermlineLocator:
    """Places a read segment in its called allele's germline (see the module docstring)."""

    def __init__(self, germlines: dict[str, str]):
        """``germlines``: {allele_name -> uppercase ungapped germline}."""
        self._germlines = germlines

    # ------------------------------------------------------------------ scoring
    @staticmethod
    def _anchor_cost(query_kmer: str, germline_window: str) -> float:
        """Directional match cost of aligning ``query_kmer`` to ``germline_window`` position-by-position:
        matches reward, mismatches penalize, and runs of the same outcome are amplified by a growing
        velocity term. Lower is a better anchor."""
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
    def _ends_are_clean(segment: str, germline: str) -> bool:
        """True iff the first and last ``_CLEAN_END`` bases each hold at most ``_CLEAN_END_MISMATCHES`` mismatches."""
        k = _CLEAN_END
        head = sum(a != b for a, b in zip(segment[:k], germline[:k]))
        tail = sum(a != b for a, b in zip(segment[-k:], germline[-k:]))
        return head <= _CLEAN_END_MISMATCHES and tail <= _CLEAN_END_MISMATCHES

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
        k = max(1, min(_ANCHOR_K, seg_len))
        len_diff = abs(ref_len - seg_len)
        # the indel head undercounts (reads with three indels can predict ~0.03), so both searches below reach at
        # least _INDEL_REACH bases past what the length alone allows
        indels = indel_allowance(indel_count)

        # 1) anchor on the END k-mer; slide the germline 3' end over the offset window. The window
        #    spans the length difference plus the indel allowance (insertions make a read longer than the
        #    germline stretch it covers, so its 3' end can sit deeper than the length difference), the
        #    deeper offsets only while the window stays inside the germline, and at most _SEARCH_SPAN deep.
        span = min(len_diff + max(indels, _INDEL_REACH), max(len_diff, ref_len - k), _SEARCH_SPAN)
        end_kmer = segment[-k:]
        best_end, best_cost = ref_len, float("inf")
        for offset in range(span + 1):
            window = germline[ref_len - (k + offset): ref_len - offset]
            cost = self._anchor_cost(end_kmer, window) + _END_OFFSET_PENALTY * offset
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
        """Substitution rate of the segment's last ``_ANCHOR_K`` bases against the germline ending at ``ref_end``: SHM
        rate where the tail is germline, ~random where it is overhang (0 when there is no tail to compare)."""
        k = min(_ANCHOR_K, len(segment), ref_end)
        if k == 0:
            return 0.0
        return sum(a != b for a, b in zip(segment[-k:], germline[ref_end - k:ref_end])) / k

    # ------------------------------------------------------------------ public API
    def locate(self, sequence: str, seq_start: int, seq_end: int, allele: str,
                  indel_count) -> GermlineMatch:
        """Where ``sequence[seq_start:seq_end]``, called ``allele`` with ``indel_count`` indels, sits in its
        germline."""
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
        if self._start_mismatch_rate(segment, germline, ref_start) > _FALLBACK_RATE:
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
                self._tail_mismatch_rate(segment, germline, ref_end) > _FALLBACK_RATE):
            seq_start = max(0, seq_start + ref_start)
            seq_end = min(len(sequence), seq_end - (ref_len - ref_end))
            ref_start, ref_end = 0, ref_len
        return GermlineMatch(seq_start, seq_end, ref_start, ref_end)
