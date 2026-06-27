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
    """Best available aligner. Tries WFA (pywfa) first when prefer=='wfa', else parasail."""
    if prefer == "wfa":
        try:
            from .wfa import WFAAligner
            return WFAAligner()
        except Exception:
            pass
    from .parasail import ParasailAligner
    return ParasailAligner()
