"""Segmentation correction: raw float boundaries -> final integer read-frame coordinates.

De-pad (per the *actual* configured ``max_len`` — TF legacy hardcoded 576, a bug the new TF pipeline
fixed), floor, clip to the read, enforce a non-empty span, then a one-directional ordering repair
(``v_end <= d_start <= d_end <= j_start <= j_end``) that only pushes *later* segments forward.

``pad_mode``: ``"right"`` (our trainer pads on the right -> coords already in the read frame, no
shift) or ``"center"`` (TF symmetric center padding -> subtract ``(max_len - seq_len)//2``).
"""
from __future__ import annotations

import numpy as np

from .state import Segments


def correct_segments(start: dict, end: dict, seq_lens: np.ndarray, max_len: int,
                     pad_mode: str = "right") -> Segments:
    seq_lens = np.asarray(seq_lens)
    pad = (max_len - seq_lens) // 2 if pad_mode == "center" else 0

    out_s, out_e = {}, {}
    for g in start:
        s = np.floor(np.asarray(start[g])).astype(int) - pad
        e = np.floor(np.asarray(end[g])).astype(int) - pad
        s = np.clip(s, 0, seq_lens - 1)
        e = np.clip(e, 1, seq_lens)
        e = np.maximum(e, s + 1)                       # non-empty end-exclusive interval
        out_s[g], out_e[g] = s, e

    # one-directional ordering repair (clamp later segments forward only)
    left = out_e["v"]
    if "d" in start:
        out_s["d"] = np.maximum(out_s["d"], left)
        out_e["d"] = np.maximum(out_e["d"], out_s["d"] + 1)
        left = out_e["d"]
    out_s["j"] = np.maximum(out_s["j"], left)
    out_e["j"] = np.maximum(out_e["j"], out_s["j"] + 1)
    return Segments(out_s, out_e)
