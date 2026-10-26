"""Stage-1 of the Gate-3 applicability detector: a HIGH-SPECIFICITY STRUCTURAL PREFILTER.

Owner decision (2026-07-24): Stage-1 hard-rejects only *decisive* structural defects and DEFERS every
uncertain case to Stage-2. A missing or SHM-mutated Cys/FR4 motif must NOT independently reject
(`WGQG->GGQG` is biological divergence, not insufficient evidence) — that ruled out ~12% of complete VHs
under the earlier rule gate. Stage-2 (learned) decides whether a deferred input is (a) a complete V domain
and (b) in the supported human-IGH-VH manifold.

Hard rejects (each validated to reject ~0 real+simulator positives; any rule that can't is demoted to
defer-only):
  - ``too_short``                            : below any complete V-domain length
  - ``reversed_order``                       : an unambiguous reversed FR4 tail at the N-terminus
  - ``leader_or_constant_contamination``     : a hydrophobic signal-peptide N-terminus, or a length that
                                               can only be a V-domain plus a constant domain
Everything else -> DEFER (``rejected=False``) to Stage-2. A hard reject maps to ``insufficient_evidence``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

RULESET_VERSION = "prefilter_v3_high_specificity"

_HYDROPHOBIC = set("ACFILMVWY")
_REV_START = re.compile(r"^[SAT]{1,2}[VITL][TL]V")     # reversed FR4 tail (…VTVSS -> SSVTV…) at the N-term

MIN_COMPLETE = 90      # below this, no complete V domain (CDR3-only ~15, J-only ~11, clear fragments)
MAX_COMPLETE = 150     # above this, a bare V domain is implausible -> a constant domain follows
LEADER_WINDOW = 18
LEADER_HYDRO_FRAC = 0.60


@dataclass
class PrefilterResult:
    rejected: bool                 # True = decisive structural reject (-> insufficient_evidence)
    reason: str | None             # set iff rejected; None means DEFER to Stage-2
    ruleset_version: str = RULESET_VERSION
    detail: dict = field(default_factory=dict)


def _looks_like_signal_peptide(seq: str) -> bool:
    """A secretion signal peptide is a short, strongly hydrophobic N-terminus lacking the FR1 disulfide
    Cys. A germline VH FR1 (EVQLVESGGG…) is far less hydrophobic and carries the FR1 Cys ~pos 22."""
    w = seq[:LEADER_WINDOW]
    hydro = sum(1 for a in w if a in _HYDROPHOBIC) / max(len(w), 1)
    return hydro >= LEADER_HYDRO_FRAC and "C" not in seq[:12]


def structural_prefilter(sequence: str) -> PrefilterResult:
    seq = sequence.strip().upper()
    n = len(seq)
    if n < MIN_COMPLETE:
        return PrefilterResult(True, "too_short", detail={"length": n})
    if _REV_START.match(seq):
        return PrefilterResult(True, "reversed_order")
    if n > MAX_COMPLETE:
        return PrefilterResult(True, "leader_or_constant_contamination", detail={"kind": "constant", "length": n})
    if _looks_like_signal_peptide(seq):
        return PrefilterResult(True, "leader_or_constant_contamination", detail={"kind": "leader"})
    return PrefilterResult(False, None)     # DEFER to Stage-2
