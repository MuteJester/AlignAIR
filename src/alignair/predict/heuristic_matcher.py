"""Heuristic germline matcher — a clean reimplementation of the TF ``HeuristicReferenceMatcher``.

Instead of a free global alignment (which drifts on mutated/truncated ends), this *anchors* on what
the model already knows. For each read segment it:

  1. anchors on the segment's **end** k-mer — V/J 3' ends are conserved — and slides a small offset
     window at the germline 3' end to find the best germline end position;
  2. **derives the start** from the known segment length, then refines it within a window whose width
     is bounded by the model's predicted **indel count**.

It is ``O(search_span · k)`` per segment — no dynamic programming — and returns germline coordinates
(no CIGAR; substitutions/indels are recovered downstream from the coordinates + the model's heads).

This is a faithful, behaviour-identical port of the TF logic, restructured for clarity: a typed
``GermlineMatch`` result, single-segment ``match_one`` split into named steps, and a ``from_reference``
constructor. See ``tests/alignair/predict/test_heuristic_matcher.py`` for the identical-results guard.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..align.backend import AlignResult

_ACCEL = 0.05          # per-step velocity increment that amplifies same-state (match/mismatch) runs


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

    Gap-free windows are a single ``M`` run (no DP). Otherwise a global alignment restricted to a
    narrow band (width = net length difference + predicted indel count + margin) places the I/D ops;
    ``mismatch``/``gap`` favour substitutions (SHM) over indels. This is the reference implementation
    and the fallback when the compiled :mod:`_derive_cy` kernel is unavailable — byte-identical output.
    """
    m, n = len(read), len(germ)
    if m == n and indels == 0:
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
except Exception:                                           # pragma: no cover - pure-Python fallback
    _derive_cigar = _derive_cigar_py
    DERIVE_BACKEND = "python"


def derive_alignment(read_window: str, germline_window: str, indel_count=0) -> AlignResult:
    """Reconstruct a base-level alignment of a read segment against its (already located) germline
    window -> an ``AlignResult`` (M/I/D CIGAR + coords), a drop-in for the WFA result.

    The window is *pinned* by :class:`HeuristicGermlineMatcher` (both ends fixed), so it can't drift
    the way an ends-free aligner does. Uses the compiled :mod:`_derive_cy` kernel when present, else
    the identical pure-Python fallback (:data:`DERIVE_BACKEND` reports which)."""
    m, n = len(read_window), len(germline_window)
    indels = max(0, int(round(_scalar(indel_count))))
    return AlignResult(0.0, _derive_cigar(read_window, germline_window, indels), 0, m, 0, n)


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
        truncated. Set to 0 to disable the fallback (pure end-anchor)."""
        self._germlines = germlines
        self._anchor_k = anchor_k
        self._search_span = search_span
        self._end_offset_penalty = end_offset_penalty
        self._fallback_rate = fallback_rate

    @classmethod
    def from_reference(cls, reference_set, **kwargs) -> "HeuristicGermlineMatcher":
        germlines = {}
        for ref in reference_set.genes.values():
            germlines.update({name: seq.upper() for name, seq in zip(ref.names, ref.sequences)})
        return cls(germlines, **kwargs)

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
        k = self._anchor_k
        seg_len, ref_len = len(segment), len(germline)
        len_diff = abs(ref_len - seg_len)

        # 1) anchor on the END k-mer; slide the germline 3' end over the offset window. The window
        #    spans the full length difference (a read can't be truncated by more than that), optionally
        #    capped by ``search_span`` — the cap is what blinded the TF version to J 3'-truncation.
        span = len_diff if self._search_span is None else min(len_diff, self._search_span)
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
        indels = max(0, int(round(float(_scalar(indel_count)))))
        if indels > 0:
            reach = min(indels, len_diff)
            offsets = range(-reach - 1, reach + 1)
        else:
            offsets = range(-1, 1)
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
    def _placement_mismatch_rate(segment: str, germline: str, ref_start: int, ref_end: int) -> float:
        """Substitution rate of the segment against the germline window the anchor chose — a sanity
        check on the placement (a correct one mismatches only at SHM rate, a mis-anchor ~randomly)."""
        window = germline[ref_start:ref_end]
        m = min(len(segment), len(window))
        if m == 0:
            return 1.0
        return sum(a != b for a, b in zip(segment, window)) / m

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

        # verify the anchor's placement; a 3'-truncated amplicon breaks the end-anchor's START
        # derivation (start = end - seg_len over-/under-shoots when the read overhangs the germline),
        # so if the chosen window mismatches far above SHM rate, relocate the START with the
        # truncation-agnostic whole-germline scan while KEEPING the anchor's end (which stays accurate).
        if self._fallback_rate and (
                self._placement_mismatch_rate(segment, germline, ref_start, ref_end) > self._fallback_rate):
            scan_start, _ = self._scan_locate(segment, germline)
            if scan_start < ref_end:
                ref_start = scan_start

        # a pure overhang (extra read bases that can't be biological indels) means the germline was
        # trimmed only because the read stuck out -> fold that trim back into the read coordinates and
        # report a full-germline span.
        pure_overhang = orig_len > ref_len or (orig_len == ref_len and indel_count == 0)
        if pure_overhang and (ref_start > 0 or ref_end < ref_len):
            seq_start = max(0, seq_start + ref_start)
            seq_end = min(len(sequence), seq_end - (ref_len - ref_end))
            ref_start, ref_end = 0, ref_len
        return GermlineMatch(seq_start, seq_end, ref_start, ref_end)

    def match(self, sequences, seq_starts, seq_ends, alleles, indel_counts) -> list[GermlineMatch]:
        """Batch ``match_one`` over aligned iterables (one segment per read for a single gene)."""
        return [self.match_one(s, ss, se, a, ic)
                for s, ss, se, a, ic in zip(sequences, seq_starts, seq_ends, alleles, indel_counts)]


def _scalar(x) -> float:
    """Coerce a possibly-numpy / 0-d indel count to a Python float (matches the TF squeeze/round)."""
    try:
        import numpy as np
        return float(np.asarray(x).squeeze())
    except Exception:
        return float(x)
