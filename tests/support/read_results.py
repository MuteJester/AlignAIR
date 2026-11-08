"""Typed reads for tests (``make_read``): a ``ReadResult`` from the few fields a test sets, each gene's call one
allele with likelihood 1."""
from __future__ import annotations

from collections.abc import Mapping

from alignair.predict.results import Anchors, GeneResult, GeneStatus, ReadResult
from alignair.predict.state import GeneCall, GermlineAlignment

_D_STATUS = {"absent": GeneStatus.ABSENT, "short": GeneStatus.ABSTAINED, "callable": GeneStatus.CALLED}


def make_read(sequence: str, *, calls: Mapping[str, str] = {}, alignments: Mapping[str, GermlineAlignment] = {},
              genes=("v", "j"), anchors: Anchors | None = None, d_presence: str | None = None, v_present: bool = True,
              indel_count: float = 0.0, mutation_rate: float = 0.0, low_quality: bool = False, locus: str | None = None,
              orientation_id: int = 0, input_sequence: str | None = None, cropped: bool = False) -> ReadResult:
    """A read of ``genes``: each gene called ``calls[g]`` (no call when absent from ``calls``) with ``alignments[g]``;
    V absent unless ``v_present``; D gated by ``d_presence`` (a ``core.d_presence.D_CLASSES`` name)."""
    results = {}
    for g in genes:
        name = calls.get(g)
        call = GeneCall((name,), (1.0,)) if name else GeneCall((), ())
        status = GeneStatus.CALLED
        if g == "v" and not v_present:
            status = GeneStatus.ABSENT
        elif g == "d" and d_presence is not None:
            status = _D_STATUS[d_presence]
        results[g] = GeneResult(call, alignments.get(g), call.names[:1], status)
    return ReadResult(sequence=sequence, input_sequence=sequence if input_sequence is None else input_sequence,
                      orientation_id=orientation_id, mutation_rate=mutation_rate, indel_count=indel_count,
                      genes=results, anchors=anchors, d_presence=d_presence, low_quality=low_quality,
                      cropped=cropped, locus=locus)
