"""Stage-1 of the Gate-3 applicability detector: a rule-based COMPLETENESS / STRUCTURE gate.

Decides whether an input is a *complete human VH domain in N->C order* before the Stage-2 membership
score is ever consulted. The rules are derived from the **domain contract** of an Ig V-domain (conserved
FR1 Cys, FR3 Cys, FR4 Trp-Gly landmarks; plausible domain length; no leader/constant flanks) — NOT
reverse-engineered from the cohort's negatives. It never trims or repairs; it accepts or rejects with a
specific reason, so the caller can withhold V/D/J calls with an honest explanation.

Reasons: ``too_short``, ``reversed_order``, ``missing_v_anchor``, ``missing_j_anchor``,
``leader_or_constant_contamination``, ``low_domain_fraction``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

RULESET_VERSION = "completeness_v1"

_CYS = re.compile(r"[YFH][YFH]C")                       # FR3-end conserved Cys (before CDR3)
# FR4 detection must survive SHM. Primary anchor: the FR4 Trp-Gly (WGxG, tolerating WGxxG) sitting near
# the C-terminus (the variable domain ends ~11 residues later). Fallback for a mutated FR4 Trp: the
# highly conserved C-terminal tail V[TL]VSS. A WGxG far from the C-terminus (constant domain following)
# is deliberately NOT a C-terminal FR4 -> such input is caught as contamination via the trailing tail.
_FR4_WG = re.compile(r"WG.{1,3}G")
_FR4_TAIL = re.compile(r"[VLT][TL]VS[SA]")
_FR4_CTERM_WINDOW = 20    # a complete VH's FR4 Trp sits within ~20 residues of the C-terminus
_FR4_SPAN = 11            # the domain ends ~11 residues after the FR4 Trp-Gly
_FR1 = re.compile(r"[QE][VILA][QKE][LV]")               # FR1 opening (EVQL/QVQL/QLQL/... ) at the N-term

MIN_DOMAIN = 90        # a complete VH variable domain is ~95-140 aa
MAX_DOMAIN = 150       # beyond this, a bare VH is implausible -> leader and/or constant present
MAX_TRAIL = 6          # residues allowed after the FR4 domain end before it reads as VH+constant
MAX_LEADER = 5         # the FR1 opening should sit at the N-term; further in implies a leader prefix
MIN_DOMAIN_FRACTION = 0.85


@dataclass
class CompletenessResult:
    passed: bool
    reason: str | None
    ruleset_version: str = RULESET_VERSION
    detail: dict = field(default_factory=dict)


def _fail(reason, **detail):
    return CompletenessResult(False, reason, RULESET_VERSION, detail)


def _fr4_end(seq: str):
    """End index of the FR4 domain, or None. Primary: the last FR4 Trp-Gly within the C-terminal window
    (domain ends ~``_FR4_SPAN`` later). Fallback (mutated Trp): the last conserved V[TL]VSS tail."""
    wg = [m for m in _FR4_WG.finditer(seq) if m.start() >= len(seq) - _FR4_CTERM_WINDOW]
    if wg:
        return min(len(seq), wg[-1].start() + _FR4_SPAN)
    tail = list(_FR4_TAIL.finditer(seq))
    return tail[-1].end() if tail else None


def check_completeness(sequence: str) -> CompletenessResult:
    seq = sequence.strip().upper()
    n = len(seq)
    if n < MIN_DOMAIN:
        return _fail("too_short", length=n)

    fr4_end = _fr4_end(seq)
    if fr4_end is None:
        # no forward FR4 — is it a reversed domain? (the reverse read carries the FR4 Trp-Gly + FR3 Cys)
        rev = seq[::-1]
        if _FR4_WG.search(rev) and _CYS.search(rev):
            return _fail("reversed_order")
        return _fail("missing_j_anchor")

    # V anchor: the invariant FR3 Cys (position ~96) sits ~10-35 residues before the FR4 end. Require a
    # Cys in that window rather than the aromatic-aromatic-Cys motif, which SHM readily disrupts.
    if "C" not in seq[max(0, fr4_end - 40):max(0, fr4_end - 6)]:
        return _fail("missing_v_anchor")

    # leader / constant contamination
    trailing = n - fr4_end
    if trailing > MAX_TRAIL:
        return _fail("leader_or_constant_contamination", trailing=trailing, kind="constant")
    fr1 = _FR1.search(seq[:35])
    domain_start = fr1.start() if fr1 else 0            # a leader pushes the FR1 opening off the N-term
    if domain_start > MAX_LEADER:
        return _fail("leader_or_constant_contamination", leader_len=domain_start, kind="leader")
    if n > MAX_DOMAIN:
        return _fail("leader_or_constant_contamination", length=n)

    # the recognized V-domain (FR1 opening .. FR4 end) must dominate the input
    if (fr4_end - domain_start) / n < MIN_DOMAIN_FRACTION:
        return _fail("low_domain_fraction", domain_span=fr4_end - domain_start, length=n)

    return CompletenessResult(True, None, RULESET_VERSION,
                              {"length": n, "fr4_end": fr4_end, "domain_start": domain_start})
