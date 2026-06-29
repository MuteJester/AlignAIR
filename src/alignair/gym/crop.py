"""Junction-centered fragment cropping for the gym.

Trains the model to classify from arbitrarily small reads (down to ~CDR3+flanks)
without assuming any input distribution: a crop is a sub-window of the simulated
read with ALL coordinates (in-sequence + germline) recomputed for the window.

Invariant: every crop window fully contains D (when present) and retains at least
``FLANK`` bp of V's 3' end and J's 5' end, so V/J/D are always present in the
target bundle (no segment is ever dropped). ``target_len`` is a floor that is
expanded as needed to satisfy this invariant.
"""

FLANK = 3
_GENES = ("v", "d", "j")


def crop_record(record: dict, target_len: int) -> dict:
    """Return a new record cropped to a junction-centered window of ~``target_len``
    bp. Returns ``record`` unchanged when ``target_len`` is None or already covers
    the read. All gene coordinates are recomputed; no gene is dropped."""
    seq = str(record["sequence"])
    L = len(seq)
    if target_len is None or target_len >= L:
        return record

    vs, ve = int(record["v_sequence_start"]), int(record["v_sequence_end"])
    js, je = int(record["j_sequence_start"]), int(record["j_sequence_end"])
    has_d = record.get("d_sequence_start") is not None
    if has_d:
        ds, de = int(record["d_sequence_start"]), int(record["d_sequence_end"])

    center = (ve + js) // 2
    half = target_len // 2
    c0 = center - half
    c1 = center + (target_len - half)

    # keep a flank of V's 3' end and J's 5' end; fully include D
    c0 = min(c0, ve - FLANK)
    c1 = max(c1, js + FLANK)
    if has_d:
        c0 = min(c0, ds)
        c1 = max(c1, de + 1)
    c0 = max(0, c0)
    c1 = min(L, c1)

    new = dict(record)
    new["sequence"] = seq[c0:c1]
    for g in _GENES:
        if record.get(f"{g}_sequence_start") is None:
            continue
        ss, ee = int(record[f"{g}_sequence_start"]), int(record[f"{g}_sequence_end"])
        gs, ge = int(record[f"{g}_germline_start"]), int(record[f"{g}_germline_end"])
        left = max(0, c0 - ss)    # gene bases lost off the 5' end
        right = max(0, ee - c1)   # gene bases lost off the 3' end
        new[f"{g}_sequence_start"] = max(0, ss - c0)
        new[f"{g}_sequence_end"] = min(c1, ee) - c0
        new[f"{g}_germline_start"] = gs + left
        new[f"{g}_germline_end"] = ge - right
    return new


def anchor_c0(record: dict, anchor) -> int:
    """Read-coordinate crop start c0 for a one-sided (3'-keep) crop.
    ("v_germline", g_start): start where V reaches germline position g_start (an FR primer site).
    ("j", keep_len): keep the 3'-most keep_len bp (a J-anchored amplicon)."""
    seq = str(record["sequence"]); L = len(seq)
    kind, val = anchor
    if kind == "j":
        return max(0, L - int(val))
    if kind == "v_germline":
        vs = int(record["v_sequence_start"]); vgs = int(record["v_germline_start"])
        return min(L, max(0, vs + max(0, int(val) - vgs)))
    raise ValueError(f"unknown anchor {anchor!r}")


def crop_one_sided(record: dict, c0: int) -> dict:
    """Keep the window [c0, len] (cut the 5' end, retain CDR3 + J). Recompute all gene coords;
    a gene whose read span lies entirely before c0 is dropped (set to None) — NO has-D / V-tail
    invariant (an adaptive read legitimately loses its 5' V)."""
    seq = str(record["sequence"]); L = len(seq)
    c0 = max(0, min(int(c0), L))
    if c0 == 0:
        return record
    new = dict(record)
    new["sequence"] = seq[c0:L]
    for g in _GENES:
        if record.get(f"{g}_sequence_start") is None:
            continue
        ss, ee = int(record[f"{g}_sequence_start"]), int(record[f"{g}_sequence_end"])
        if ee <= c0:                                          # gene entirely 5' of the window -> absent
            for k in (f"{g}_sequence_start", f"{g}_sequence_end",
                      f"{g}_germline_start", f"{g}_germline_end", f"{g}_call"):
                new[k] = None
            continue
        gs, ge = int(record[f"{g}_germline_start"]), int(record[f"{g}_germline_end"])
        left = max(0, c0 - ss)                                # gene bases lost off the 5' end
        new[f"{g}_sequence_start"] = max(0, ss - c0)
        new[f"{g}_sequence_end"] = ee - c0
        new[f"{g}_germline_start"] = gs + left
        new[f"{g}_germline_end"] = ge
    return new
