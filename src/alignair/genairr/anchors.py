"""V and J anchors as GenAIRR's resolver confirms them.

GenAIRR's bundled dataconfigs store ``anchor = 0`` for an allele whose conserved codon (V Cys104, J Trp/Phe118) was not
found, a legacy "not found" that reads as a real position. ``confirmed_anchor`` keeps every other stored anchor and
asks GenAIRR's native resolver about a stored 0 (or none): V from its IMGT-gapped form (column 309), J from a unique
F/W-G-X-G motif. The resolver is called on the allele's sequences, never through
``GenAIRR._reference_annotations.resolve_allele_anchor``, which writes its result into the dataconfig's allele."""
from __future__ import annotations

import json

_CONFIRMED = ("inferred", "verified")


def _resolve(allele, segment: str) -> dict:
    from GenAIRR import _engine
    gapped = getattr(allele, "gapped_seq", None) if segment == "V" else None
    return json.loads(_engine.resolve_reference_anchor(
        allele.ungapped_seq.encode("ascii"), segment, locus=allele.name[:3].upper(),
        gapped_sequence=gapped.encode("ascii") if gapped else None, imgt_gapped=bool(gapped)))


def confirmed_anchor(allele, segment: str) -> int | None:
    """``allele``'s anchor (an ungapped germline position) for ``segment`` ``"V"`` or ``"J"``: a stored non-zero anchor,
    else the resolver's position when it confirms one, else None."""
    stored = getattr(allele, "anchor", None)
    if stored:
        return int(stored)
    result = _resolve(allele, segment)
    return int(result["position"]) if result["status"] in _CONFIRMED and result["position"] is not None else None
