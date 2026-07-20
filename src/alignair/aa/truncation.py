"""Coordinate-aware truncation for AA-native partial-coverage training.

A crop keeps a contiguous residue window of the read; all V/D/J boundaries are shifted into crop
coordinates and clipped, a gene whose region falls entirely outside the window is dropped (masked
from the loss), and a partially observed gene is supervised against the COMPATIBLE protein-class set
(coverage-aware ambiguity) rather than a single unknowable label."""
from __future__ import annotations


def crop_window(n_res: int, kind: str, keep: int, start: int | None = None):
    """The kept window [lo, hi) of a read of ``n_res`` residues. ``kind``: nterm (prefix), cterm
    (suffix), internal (a middle window, at ``start`` if given)."""
    keep = max(1, min(keep, n_res))
    if kind == "nterm":
        return 0, keep
    if kind == "cterm":
        return n_res - keep, n_res
    lo = start if start is not None else max(0, (n_res - keep) // 2)
    lo = max(0, min(lo, n_res - keep))
    return lo, lo + keep


def shift_clip(boundary, lo: int, hi: int):
    """Map a half-open gene span into crop coords [0, hi-lo); return None if it is entirely outside."""
    s, e = boundary
    ns, ne = max(s, lo) - lo, min(e, hi) - lo
    return (ns, ne) if ne > ns else None


def compatible_classes(true_class: int, ga: int, gb: int, class_protos: dict) -> set:
    """Protein classes indistinguishable from ``true_class`` over germline residues [ga, gb) — i.e.
    those whose prototype matches the true prototype on the observed window. Grows as the window
    shrinks (coverage-aware ambiguity). Assumes prototypes share the V-start coordinate frame."""
    ref = class_protos.get(true_class, "")
    key = ref[ga:gb]
    return {c for c, p in class_protos.items() if p[ga:gb] == key}
