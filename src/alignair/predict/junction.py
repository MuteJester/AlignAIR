"""The junction, reading frame, stop codon and productivity of a read, derived by rule from its V and J anchors.

The V anchor is the first base of the conserved V Cys codon; the J anchor is the first base of the conserved J
Trp/Phe codon. The junction runs from the V anchor to the end of the J anchor's codon (``j_anchor + 3``), as AIRR
defines it: the CDR3 plus the two conserved codons. Productivity is not learned; it follows from the anchors and the
read's coding span.

**The rule** is GenAIRR's ``canonical_observed_vdj_v1`` productivity policy as the AlignAIR assay's truth reads it
(``alignair_assay.truth.build``), on the read the model saw, in its forward frame, with 0-based, end-exclusive
coordinates:

- **Junction.** It exists when both anchors are observed (their codons lie wholly inside the read) and the J
  anchor's codon ends after the V anchor (``j_anchor + 3 > v_anchor``). Otherwise the junction, the frame, the stop
  and the productivity are all None (unknown). An anchor end loss removed is not observed, so a clipped read is
  unknown, never given a coordinate outside it.
- **Frame.** ``vj_in_frame`` is ``(junction_end - junction_start) % 3 == 0``.
- **Stop.** ``stop_codon`` is True when a stop codon (TAA, TAG, TGA) lies in the junction's reading frame anywhere
  in the coding span ``[v_start, j_end)``: the read's V start to its J end, so a stop in V or J outside the junction
  counts. The codons are read from the span's first base on the junction's frame; a codon counts only when it lies
  wholly inside the span (and the read). It is searched whether or not the junction is in frame, as the assay's truth
  searches it (which scores it only on in-frame reads).
- **Productive.** ``vj_in_frame and not stop_codon``.

GenAIRR's own record goes further than this rule. It checks the V coding frame from the V germline start, the anchor
residues and recombination trims, and calls a read unproductive with no junction when it sees a stop. Those need
germline knowledge an anchor does not carry, and the assay's truth does not use them. A stop without a junction is
None here, as in the assay's truth. The assay's truth also calls a read with a V or J frameshift (net indel not a
multiple of 3) unproductive; this rule does not, since the anchors cannot see an indel.

**Unknown span.** Without the read's V start or J end (``coding_span`` None, or either end None) the stop is
unknown: an in-frame junction's productivity is then unknown, and an out-of-frame one is unproductive.

On reads in the product's ACGTN alphabet, these three codons are exactly the codons the assay's IUPAC-aware
translation reads as a stop: no codon holding an N is a stop for every base N may stand for.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index

from ..core.anchors import ANCHOR_KEYS, CODON
from ..errors import InputError
from ..reference.codons import STOP_CODONS


@dataclass(frozen=True)
class JunctionCall:
    """A read's junction ``[junction_start, junction_end)`` (0-based, end-exclusive, read coordinates) and its
    frame, stop codon and productivity; every field is None when the junction is unknown."""
    junction_start: int | None
    junction_end: int | None
    vj_in_frame: bool | None
    stop_codon: bool | None
    productive: bool | None


UNKNOWN = JunctionCall(None, None, None, None, None)


def derive_junction(sequence: str, v_anchor: int | None, j_anchor: int | None, *,
                    coding_span: tuple[int | None, int | None] | None) -> JunctionCall:
    """The junction, frame, stop codon and productivity of ``sequence`` given its anchors and its coding span
    ``(v_start, j_end)`` (see the module docstring for the rule). An anchor is None when it is not observed."""
    v_anchor, j_anchor = _anchor(v_anchor, ANCHOR_KEYS["v"]), _anchor(j_anchor, ANCHOR_KEYS["j"])
    n = len(sequence)
    if v_anchor is None or j_anchor is None or not (_codon_in(v_anchor, n) and _codon_in(j_anchor, n)):
        return UNKNOWN
    start, end = v_anchor, j_anchor + CODON
    if not end > start:
        return UNKNOWN
    in_frame = (end - start) % CODON == 0
    stop = _stop_in_span(sequence, start, coding_span)
    if stop is None:
        return JunctionCall(start, end, in_frame, None, False if not in_frame else None)
    return JunctionCall(start, end, in_frame, stop, in_frame and not stop)


def _anchor(value, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise InputError(f"{field} must be an integer read position or None", context={"field": field})
    try:
        return index(value)
    except TypeError as exc:
        raise InputError(f"{field} must be an integer read position or None", context={"field": field}) from exc


def _codon_in(position: int, length: int) -> bool:
    return 0 <= position and position + CODON <= length


def _stop_in_span(sequence: str, junction_start: int, coding_span) -> bool | None:
    """Whether a stop codon lies in the junction's frame inside the coding span (clipped to the read); None when
    the span is unknown."""
    if coding_span is None:
        return None
    v_start, j_end = coding_span
    if v_start is None or j_end is None:
        return None
    v_start, j_end = max(0, index(v_start)), min(len(sequence), index(j_end))
    first = v_start + (junction_start - v_start) % CODON
    read = sequence.upper()
    return any(read[k:k + CODON] in STOP_CODONS for k in range(first, j_end - CODON + 1, CODON))
