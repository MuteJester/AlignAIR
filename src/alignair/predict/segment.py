"""Segmentation correction: raw float boundaries -> final integer read-frame coordinates.

De-pad (per the *actual* configured ``max_len`` — TF legacy hardcoded 576, a bug the new TF pipeline
fixed), floor, then a **constrained projection** that enforces every invariant *jointly* rather than a
one-directional repair (which could push later segments past the read):

  * bounds: ``0 <= start <= end <= seq_len`` for every segment;
  * ordering: ``v_end <= d_start <= d_end <= j_start <= j_end`` (light chains skip D);
  * a preferred non-empty span (``min_span``) for each present segment, but only *if it fits* — a
    segment squeezed out by a too-short read collapses to a zero-length (absent) interval rather than
    manufacturing sequence outside the read.

The projection walks the ordered boundary points left-to-right as a non-decreasing chain capped at
``seq_len``: each boundary is clamped to ``[previous_boundary, seq_len]``. This makes bounds and
ordering hold by construction, and a collapse of the mandatory V anchor is reported via
``Segments.low_quality`` (no feasible layout for that read).

``pad_mode``: ``"right"`` (our trainer pads on the right -> coords already in the read frame, no
shift) or ``"center"`` (TF symmetric center padding -> subtract ``(max_len - seq_len)//2``).
"""
from __future__ import annotations

import numpy as np

from .state import Segments


def correct_segments(start: dict, end: dict, seq_lens: np.ndarray, max_len: int,
                     pad_mode: str = "right", min_span: int = 1) -> Segments:
    seq_lens = np.asarray(seq_lens).astype(int)
    pad = (max_len - seq_lens) // 2 if pad_mode == "center" else 0
    genes = [g for g in ("v", "d", "j") if g in start]

    # 1) per-segment: de-pad, floor, clip into [0, L], and a *bounded* min-span preference that can
    #    never exceed L (so it is overridden, not enforced, by the ordering projection below).
    s_pref, e_pref = {}, {}
    for g in genes:
        s = np.clip(np.floor(np.asarray(start[g])).astype(int) - pad, 0, seq_lens)
        e = np.clip(np.floor(np.asarray(end[g])).astype(int) - pad, 0, seq_lens)
        e = np.minimum(np.maximum(e, s + min_span), seq_lens)      # prefer >=min_span, capped at L
        s_pref[g], e_pref[g] = s, e

    # 2) constrained projection: non-decreasing boundary chain capped at L -> bounds + ordering jointly
    out_s, out_e = {}, {}
    prev = np.zeros_like(seq_lens)
    for g in genes:
        s = np.clip(s_pref[g], prev, seq_lens)
        e = np.clip(e_pref[g], s, seq_lens)
        out_s[g], out_e[g] = s, e
        prev = e

    # a mandatory framework segment (V or J) collapsing to zero length => no feasible layout for the
    # read (D is optional and may legitimately be absent, so it is not counted).
    low_quality = ((out_e["v"] - out_s["v"]) <= 0) | ((out_e["j"] - out_s["j"]) <= 0)
    return Segments(out_s, out_e, low_quality=low_quality)
