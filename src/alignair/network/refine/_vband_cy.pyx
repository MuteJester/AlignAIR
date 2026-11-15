# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled kernel for the V re-score's band (``alignair.network.refine.v_rescore``).

Each candidate row's banded semi-global score against the read's span, with the pair scores computed per cell
instead of as a (B, K, T, W) tensor. Bit-for-bit the torch band (``vrescore._torch_banded_scores``): the same float32
values (``match`` and ``mismatch`` come in computed by torch) are selected, added and maxed in the same order. Built
as an optional extension; without the ``.so`` the torch band is used.
"""
from libc.stdint cimport int64_t

cdef enum:
    MAX_WIDTH = 64              # the widest band the kernel holds (today's: 17 diagonals)


def band_scores(const int64_t[:, ::1] tokens, const int64_t[::1] starts, const int64_t[::1] ends,
                const int64_t[::1] anchors, const int64_t[:, :, ::1] rows, const int64_t[:, ::1] row_anchors,
                const float[::1] match, const float[::1] mismatch, int band, float gap, float outside,
                float unaligned, float[:, ::1] out):
    """Write ``out[b, k]``, row k's band score against read b's span [starts[b], ends[b]). Row base
    ``t + row_anchors[b, k] - anchors[b] + w - band`` is read base t's partner on band diagonal w."""
    cdef Py_ssize_t batch = rows.shape[0], candidates = rows.shape[1], row_length = rows.shape[2]
    cdef int width = 2 * band + 1
    if width > MAX_WIDTH:
        raise ValueError(f"band {band} is wider than the kernel's {MAX_WIDTH} diagonals")
    cdef float best[MAX_WIDTH]
    cdef float two_gap = 2 * gap
    cdef float run, value, inserted, pair, top
    cdef Py_ssize_t b, k, t, w, p
    cdef int64_t base, germline, offset
    with nogil:
        for b in range(batch):
            for k in range(candidates):
                offset = row_anchors[b, k] - anchors[b] - band
                for w in range(width):
                    best[w] = gap * w
                for t in range(starts[b], ends[b]):
                    base = tokens[b, t]
                    run = best[0]
                    for w in range(width):
                        if best[w] > run:
                            run = best[w]
                        inserted = best[w + 1] - two_gap if w < width - 1 else unaligned
                        if base < 1 or base > 4:
                            pair = 0
                        else:
                            p = t + offset + w
                            germline = rows[b, k, p] if 0 <= p < row_length else 0
                            if germline <= 0:
                                pair = outside
                            elif germline == base:
                                pair = match[b]
                            else:
                                pair = mismatch[b]
                        value = run + pair
                        best[w] = value if value > inserted else inserted
                top = best[0]
                for w in range(width):
                    if best[w] - gap * w > top:
                        top = best[w] - gap * w
                out[b, k] = top
