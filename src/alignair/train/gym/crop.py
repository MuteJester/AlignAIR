"""Junction-centered fragment cropping for the gym.

Trains the model to classify from arbitrarily small reads (down to ~CDR3+flanks)
without assuming any input distribution: a crop is a sub-window of the simulated
read with ALL coordinates (in-sequence + germline) recomputed for the window.

Every crop window fully contains visible D and retains the available V 3' and
J 5' flanks (up to ``FLANK`` bases). Already absent genes remain absent.
``target_len`` is a floor expanded as needed to retain those observed flanks.
"""

from .coordinates import segment_spans

FLANK = 3
_GENES = ("v", "d", "j")


def crop_record(record: dict, target_len: int) -> dict:
    """Return a new record cropped to a junction-centered window of ~``target_len``
    bp. Returns ``record`` unchanged when ``target_len`` is None or already covers
    the read. Visible genes are retained; already absent genes stay absent."""
    seq = str(record["sequence"])
    L = len(seq)
    if target_len is None or target_len >= L:
        return record

    coords, germ = segment_spans(record)
    vs, ve = coords.get("v", (0, 0))
    js, je = coords.get("j", (L, L))
    has_d = "d" in coords
    if has_d:
        ds, de = coords["d"]

    center = (ve + js) // 2
    half = target_len // 2
    c0 = center - half
    c1 = center + (target_len - half)

    # keep a flank of V's 3' end and J's 5' end; fully include D
    if "v" in coords:
        c0 = min(c0, ve - FLANK)
    if "j" in coords:
        c1 = max(c1, js + FLANK)
    if has_d:
        c0 = min(c0, ds)
        c1 = max(c1, de + 1)
    c0 = max(0, c0)
    c1 = min(L, c1)

    new = dict(record)
    new["sequence"] = seq[c0:c1]
    for g in _GENES:
        if g not in coords:
            for field in ("sequence_start", "sequence_end", "germline_start", "germline_end"):
                key = f"{g}_{field}"
                if key in new:
                    new[key] = None
            continue
        ss, ee = coords[g]
        gs, ge = germ[g]
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
    coords, germ = segment_spans(record)
    for g in _GENES:
        if g not in coords or coords[g][1] <= c0:
            for k in (f"{g}_sequence_start", f"{g}_sequence_end",
                      f"{g}_germline_start", f"{g}_germline_end", f"{g}_call"):
                new[k] = None
            continue
        ss, ee = coords[g]
        gs, ge = germ[g]
        left = max(0, c0 - ss)                                # gene bases lost off the 5' end
        new[f"{g}_sequence_start"] = max(0, ss - c0)
        new[f"{g}_sequence_end"] = ee - c0
        new[f"{g}_germline_start"] = gs + left
        new[f"{g}_germline_end"] = ge
    return new
