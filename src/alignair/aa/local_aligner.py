"""No-training, pure-AA V caller: local (Smith-Waterman, affine-gap, BLOSUM62) alignment of the
observed protein against one prototype per V protein-equivalence class.

This is the de-risk baseline for whether alignment-based recovery is worth building into the model.
It uses ONLY the amino-acid sequence — no true junction coordinate, no nucleotide CIGAR. Local
alignment implicitly scans for the best-matching block, so under an indel it aligns the in-frame
region and gaps out the frameshifted part (which is why it can beat a global/frame-fixed read)."""
from __future__ import annotations

import parasail

_VALID = set("ACDEFGHIKLMNPQRSTVWYXBZ*")


def _sanitize(seq: str) -> str:
    return "".join(c if c in _VALID else "X" for c in seq.upper())


def v_class_prototypes(germlines: dict, class_of: dict):
    """One representative protein per V class (synonymous alleles share a protein, so any member works).
    ``germlines``: {name: AAGermline}; ``class_of``: {name: class_id}. Returns (classes, protos)."""
    proto: dict = {}
    for name, g in germlines.items():
        c = class_of.get(name)
        if c is not None and c not in proto:
            proto[c] = g.aa_seq
    classes = sorted(proto)
    return classes, [proto[c] for c in classes]


def classify_v(query: str, classes, protos, gap_open: int = 11, gap_extend: int = 1, matrix=None):
    """Return (best_class, best_score) by local alignment of ``query`` against each class prototype."""
    matrix = matrix or parasail.blosum62
    q = _sanitize(query)
    if not q:
        return (classes[0] if classes else -1), -1.0
    prof = parasail.profile_create_16(q, matrix)
    best_i, best_s = 0, -1 << 30
    for i, ref in enumerate(protos):
        s = parasail.sw_striped_profile_16(prof, ref, gap_open, gap_extend).score
        if s > best_s:
            best_s, best_i = s, i
    return classes[best_i], int(best_s)
