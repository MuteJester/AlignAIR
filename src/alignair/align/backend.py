from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AlignResult:
    score: float        # higher is better
    cigar: str          # core M/I/D run-length ops only (no S/N)
    q_start: int        # first query base consumed (0-based)
    q_end: int          # one past last query base
    t_start: int        # germline (target) bases consumed before the core
    t_end: int          # germline end (exclusive)


@runtime_checkable
class Aligner(Protocol):
    def align(self, query: str, target: str) -> AlignResult | None: ...


def get_aligner(prefer: str = "wfa") -> Aligner:
    """Best *available* aligner, probing each optional backend before selecting it:
    WFA (pywfa) when prefer=='wfa' -> parasail -> Biopython fallback.

    The probe matters: constructing a backend does not import its native dependency (that is lazy),
    so an unchecked selection would return a backend that fails only at align() time. Biopython is a
    core dependency, so this always returns a working aligner even with no optional backend installed
    (e.g. a default Apple Silicon install, where parasail has no wheel)."""
    if prefer == "wfa":
        from .wfa import wfa_available
        if wfa_available():
            from .wfa import WFAAligner
            return WFAAligner()
    from .parasail import parasail_available
    if parasail_available():
        from .parasail import ParasailAligner
        return ParasailAligner()
    from .bio import BioAligner
    return BioAligner()
