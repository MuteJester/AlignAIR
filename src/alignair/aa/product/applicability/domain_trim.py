"""Frozen, versioned domain-preparation procedures for the Gate-3 cohort cleanup gate.

These are DATA-PREP transforms for building the cohort — NOT the detector. They exist because ANARCI/IMGT
numbering was unavailable in the build environment and the SAbDab summaries carry no VH-boundary
annotation, so a documented motif-anchored fallback is used (spec Gate-3 allows ANARCI/IMGT parsing "only
as a documented fallback"). Anchors used are the most conserved landmarks of an Ig/TCR V-domain:
  - FR3 end  -> the invariant `[YFH][YFH]C` (aromatic-aromatic-Cys before CDR3), and
  - FR4 start -> the invariant `WG[A-Z]G` (Trp-Gly of FR4).
Every use records original+trimmed hashes and boundaries; anything where the anchor is absent or the
result falls outside a plausible V-domain length is DROPPED, never guessed.
"""
from __future__ import annotations

import hashlib
import re

TRIM_PROCEDURE = "motif_fr4_junction_v1"
REBUILD_PROCEDURE = "motif_cys_wgxg_v1"

# End of FR4 (the V-domain / constant junction): WGxG then, a few residues later, the domain tail V[TS]VSS.
_FR4_TAIL = re.compile(r"WG[A-Z]G.{1,8}?V[TS]VS[SAT]")
_CYS = re.compile(r"[YFH][YFH]C")     # FR3-end conserved Cys (before CDR3)
_WGXG = re.compile(r"WG[A-Z]G")       # FR4-start conserved Trp-Gly

MIN_VH, MAX_VH = 90, 160              # plausible single V-domain length (aa)


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def trim_vh(seq: str):
    """Trim a full antibody heavy / VHH chain to its variable domain at the FR4->constant junction.

    Returns ``(trimmed, end_index)`` or ``None`` when the FR4 anchor is absent or the trimmed length is
    outside ``[MIN_VH, MAX_VH]`` (drop, do not guess). Cuts after the LAST ``WGxG...V[TS]VSS`` match
    (the antibody's own FR4, before any constant domain)."""
    matches = list(_FR4_TAIL.finditer(seq))
    if not matches:
        return None
    end = matches[-1].end()
    trimmed = seq[:end]
    if not (MIN_VH <= len(trimmed) <= MAX_VH):
        return None
    return trimmed, end


def rebuild_cdr3_j(parent_vh: str):
    """From a (complete) VH amino-acid sequence, extract ``(cdr3, j_region)`` deterministically.

    ``cdr3`` = residues between the FR3-end Cys and the FR4-start Trp; ``j_region`` = FR4 onward (the
    J-only fragment). Returns ``None`` if either anchor is missing or the pieces are implausibly short.
    Used to rebuild the ``cdr3_only`` / ``j_only`` structural negatives from their parent VHs while
    preserving the parent lineage key (so transforms never inflate effective N)."""
    cys = list(_CYS.finditer(parent_vh))
    wgxg = list(_WGXG.finditer(parent_vh))
    if not cys or not wgxg:
        return None
    c = cys[-1].end()
    after = [m for m in wgxg if m.start() >= c]
    if not after:
        return None
    fr4 = after[-1].start()
    cdr3, j = parent_vh[c:fr4], parent_vh[fr4:]
    if len(cdr3) < 3 or len(j) < 5:
        return None
    return cdr3, j
