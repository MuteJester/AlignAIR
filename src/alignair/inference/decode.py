"""Decode raw model outputs into integer boundary positions, padding-corrected.

Faithful port of the legacy CleanAndExtractStage (argmax positions) and
SegmentCorrectionStage (remove center-pad offset, clamp, enforce V<=D<=J)."""
import numpy as np


def extract_positions(pred: dict, has_d: bool) -> dict:
    """Argmax of *_start_logits / *_end_logits -> integer positions (N,)."""
    genes = ["v", "j"] + (["d"] if has_d else [])
    out = {}
    for g in genes:
        out[f"{g}_start"] = np.argmax(pred[f"{g}_start_logits"], axis=-1).astype(np.int32)
        out[f"{g}_end"] = np.argmax(pred[f"{g}_end_logits"], axis=-1).astype(np.int32)
    return out


def correct_segments(positions: dict, sequences, max_length: int, has_d: bool) -> dict:
    """Remove per-sequence center padding, clamp to sequence bounds, enforce V<=D<=J."""
    paddings = np.array([(max_length - len(s)) // 2 for s in sequences], dtype=np.int32)
    seq_lengths = np.array([len(s) for s in sequences], dtype=np.int32)

    def sanitize(raw_start, raw_end):
        s = np.floor(np.squeeze(raw_start) - paddings).astype(np.int32)
        e = np.floor(np.squeeze(raw_end) - paddings).astype(np.int32)
        s = np.clip(s, 0, seq_lengths - 1)
        e = np.clip(e, 1, seq_lengths)
        e = np.maximum(e, s + 1)
        return s, e

    v_start, v_end = sanitize(positions["v_start"], positions["v_end"])
    j_start, j_end = sanitize(positions["j_start"], positions["j_end"])

    d_start = d_end = None
    if has_d and positions.get("d_start") is not None:
        d_start, d_end = sanitize(positions["d_start"], positions["d_end"])

    if d_start is not None:
        d_start = np.maximum(d_start, v_end)
        d_end = np.maximum(d_end, d_start + 1)
        j_start = np.maximum(j_start, d_end)
        j_end = np.maximum(j_end, j_start + 1)
    else:
        j_start = np.maximum(j_start, v_end)
        j_end = np.maximum(j_end, j_start + 1)

    # Final bounds clamp (improvement over legacy, which left ordering free to push
    # positions past the sequence end when an upstream boundary is at the very end —
    # common with untrained weights). Keep start in [0, L-1] and end in [start+1, L].
    def clamp(s, e):
        s = np.clip(s, 0, seq_lengths - 1)
        e = np.clip(e, s + 1, seq_lengths)
        return s, e

    v_start, v_end = clamp(v_start, v_end)
    j_start, j_end = clamp(j_start, j_end)
    if d_start is not None:
        d_start, d_end = clamp(d_start, d_end)

    corrected = {"v_start": v_start, "v_end": v_end, "j_start": j_start, "j_end": j_end}
    if d_start is not None:
        corrected["d_start"] = d_start
        corrected["d_end"] = d_end
    return corrected
