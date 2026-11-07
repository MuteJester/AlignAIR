"""V and J anchors as GenAIRR's resolver gives them: the one place AlignAIR asks GenAIRR where an allele's conserved
codon (V Cys104, J Trp/Phe118) is.

``resolve_anchor`` returns the native resolver's verdict on a sequence (V from its IMGT-gapped form, column 309; J
from a unique F/W-G-X-G motif, under its locus's rule) as a ``ResolvedAnchor``. It calls the resolver on the
sequences, never through ``GenAIRR._reference_annotations.resolve_allele_anchor``, which writes its result into a
dataconfig's allele. When GenAIRR makes the resolver public (ask G4), only ``_native`` changes.

GenAIRR's bundled dataconfigs store ``anchor = 0`` for an allele whose codon was not found, a legacy "not found" that
reads as a real position; ``confirmed_anchor`` keeps every other stored anchor and resolves a stored 0 (or none)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from ..errors import MissingDependencyError

_CONFIRMED = ("inferred", "verified")


@dataclass(frozen=True)
class ResolvedAnchor:
    """GenAIRR's verdict: ``position`` (an ungapped germline position) when ``status`` is inferred or verified; the
    status (``inferred``, ``verified``, ``unknown`` or ``invalid``), method and reason verbatim; the candidates."""
    position: int | None
    status: str
    method: str
    reason: str
    candidates: tuple


def _native():
    from GenAIRR import _engine
    return _engine


def resolve_anchor(sequence: str, segment: Literal["V", "J"], *, locus: str,
                   gapped: str | None = None) -> ResolvedAnchor:
    """GenAIRR's verdict on where ``sequence``'s anchor is: a V from its IMGT-gapped form ``gapped`` (an ungapped V
    has none), a J from a unique motif under ``locus``'s rule. ValueError for a D; MissingDependencyError when the
    installed GenAIRR has no resolver."""
    if segment not in ("V", "J"):
        raise ValueError(f"only a V or a J has an anchor, got segment {segment!r}")
    resolver = getattr(_native(), "resolve_reference_anchor", None)
    if resolver is None:
        raise MissingDependencyError("GenAIRR's anchor resolver is unavailable", context={"segment": segment},
                                     hint="Install GenAIRR>=2.4.0,<2.5.")
    result = json.loads(resolver(sequence.encode("ascii", "replace"), segment, locus=locus,
                                 gapped_sequence=gapped.encode("ascii", "replace") if gapped else None,
                                 imgt_gapped=bool(gapped)))
    position = result.get("position")
    return ResolvedAnchor(None if position is None else int(position), result["status"], result["method"],
                          result.get("reason", ""), tuple(int(c) for c in result.get("candidates") or ()))


def confirmed_anchor(allele, segment: str) -> int | None:
    """``allele``'s anchor (an ungapped germline position) for ``segment`` ``"V"`` or ``"J"``: a stored non-zero anchor,
    else the resolver's position when it confirms one, else None."""
    stored = getattr(allele, "anchor", None)
    if stored:
        return int(stored)
    gapped = getattr(allele, "gapped_seq", None) if segment == "V" else None
    result = resolve_anchor(allele.ungapped_seq, segment, locus=allele.name[:3].upper(), gapped=gapped)
    return result.position if result.status in _CONFIRMED else None
