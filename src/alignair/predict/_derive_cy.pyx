# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled C-speed kernel for the germline CIGAR reconstruction.

A banded global alignment of a *pinned* read/germline window (both ends fixed by the
``HeuristicGermlineMatcher``), producing an M/I/D CIGAR. Byte-identical to the pure-Python reference
``alignair.predict.heuristic_matcher._derive_cigar_py`` (same scoring, band and tie-breaking), ~100x
faster. Built as an optional extension; if the ``.so`` is absent the pure-Python fallback is used.
"""
from libc.stdlib cimport malloc, free


cdef str _rle(list ops):
    if not ops:
        return ""
    cdef list out = []
    cdef str p = ops[0]
    cdef int c = 1
    cdef str o
    for o in ops[1:]:
        if o == p:
            c += 1
        else:
            out.append("%d%s" % (c, p)); p = o; c = 1
    out.append("%d%s" % (c, p))
    return "".join(out)


def derive_cigar(bytes read, bytes germ, int indel_count=0,
                 int mismatch=1, int gap=2, int margin=2):
    """(read, germline) ascii bytes + predicted indel count -> M/I/D CIGAR string."""
    cdef int m = len(read)
    cdef int n = len(germ)
    cdef int indels = indel_count if indel_count > 0 else 0
    if m == n and indels == 0:
        return ("%dM" % m) if m else ""
    if m == 0:
        return ("%dD" % n) if n else ""
    if n == 0:
        return ("%dI" % m) if m else ""

    cdef const unsigned char* R = read
    cdef const unsigned char* G = germ
    cdef int d = m - n
    cdef int amd = d if d >= 0 else -d
    cdef int band = amd + indels + margin
    cdef int lo = (0 if d > 0 else d) - band
    cdef int hi = (d if d > 0 else 0) + band
    cdef int W = hi - lo + 1
    cdef double INF = 1e18

    cdef double* prev = <double*>malloc(W * sizeof(double))
    cdef double* cur = <double*>malloc(W * sizeof(double))
    cdef signed char* back = <signed char*>malloc(<size_t>(m + 1) * W * sizeof(signed char))
    if prev == NULL or cur == NULL or back == NULL:
        if prev != NULL: free(prev)
        if cur != NULL: free(cur)
        if back != NULL: free(back)
        raise MemoryError()

    cdef int i, j, k, k0 = -lo
    cdef double best, c, pd, pu, nb
    cdef signed char op

    for k in range(W):
        prev[k] = INF
        back[k] = 0
    prev[k0] = 0.0
    for j in range(1, n + 1):
        k = k0 - j
        if 0 <= k < W:
            prev[k] = j * gap
            back[k] = 3

    for i in range(1, m + 1):
        for k in range(W):
            cur[k] = INF
            op = 0
            j = i - (lo + k)
            if j < 0 or j > n:
                back[i * W + k] = 0
                continue
            best = INF
            if j >= 1:
                pd = prev[k]
                if pd < INF:
                    c = pd + (0 if R[i - 1] == G[j - 1] else mismatch)
                    if c < best:
                        best = c; op = 1
            if k >= 1:
                pu = prev[k - 1]
                if pu < INF:
                    c = pu + gap
                    if c < best:
                        best = c; op = 2
            cur[k] = best
            back[i * W + k] = op
        for k in range(W - 2, -1, -1):
            nb = cur[k + 1]
            if nb < INF:
                c = nb + gap
                if c < cur[k]:
                    cur[k] = c
                    back[i * W + k] = 3
        for k in range(W):
            prev[k] = cur[k]

    cdef list ops = []
    i = m
    k = (m - n) - lo
    cdef signed char o2
    while not (i == 0 and (i - (lo + k)) == 0):
        o2 = back[i * W + k]
        if o2 == 0:
            break
        if o2 == 1:
            ops.append("M"); i -= 1
        elif o2 == 2:
            ops.append("I"); i -= 1; k -= 1
        else:
            ops.append("D"); k += 1
    free(prev); free(cur); free(back)
    ops.reverse()
    return _rle(ops)
