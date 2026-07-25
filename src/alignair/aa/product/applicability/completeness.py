"""Stage-1 of the Gate-3 applicability detector: a rule-based COMPLETENESS / STRUCTURE gate.

**Locus-agnostic** (owner decision 2026-07-24): Stage-1 asks only "is this a complete immunoglobulin/
TCR-fold variable domain in N->C order, free of leader/constant flanks?" — NOT "is it IGH?". Locus
membership (human IGH VH vs IGK/IGL/TCR/other) is Stage-2's job. This split fixes a ~12% false-reject on
complete VHs whose FR4 Trp is SHM-divergent (they now pass Stage-1), and puts locus discrimination where
it belongs.

The universal landmarks of an Ig/TCR V-domain are the two conserved cysteines forming the intradomain
disulfide (FR1 Cys ~pos 22, FR3 Cys ~pos 96, ~50-85 apart) followed by a CDR3 and a J-region. The rules
are derived from that contract, not reverse-engineered from the cohort's negatives.

Reasons: ``too_short``, ``reversed_order``, ``missing_v_anchor``, ``missing_j_anchor``,
``leader_or_constant_contamination``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

RULESET_VERSION = "completeness_v2_locus_agnostic"

# J-region motif shared across loci: heavy FR4 WGxG, light/TCR FGxG (Trp/Phe-Gly-x-Gly), tolerating a
# 1-3 residue gap under SHM; plus the highly conserved C-terminal tail as a mutated-Trp fallback.
_JMOTIF = re.compile(r"[WFY]G.{1,3}G")
_JTAIL = re.compile(r"[VLT][TL][VK][SLA]")
_REV_START = re.compile(r"^[SAT]{1,2}[VITL][TL]V")     # a reversed FR4 tail (…VTVSS -> SSVTV…) at the N-term

MIN_DOMAIN = 90        # a complete variable domain is ~95-140 aa
MAX_DOMAIN = 150       # beyond this, a bare V-domain is implausible -> leader and/or constant present
CYS_PAIR_MIN, CYS_PAIR_MAX = 50, 88     # FR1-Cys .. FR3-Cys separation
FR1_CYS_MAX = 32       # the FR1 Cys sits ~pos 22; much later implies a leader prefix
MIN_J_TRAIL = 13       # residues after the FR3 Cys (CDR3 + a J-region's worth)
J_CTERM_WINDOW = 22    # the J motif sits within ~22 residues of the C-terminus


@dataclass
class CompletenessResult:
    passed: bool
    reason: str | None
    ruleset_version: str = RULESET_VERSION
    detail: dict = field(default_factory=dict)


def _fail(reason, **detail):
    return CompletenessResult(False, reason, RULESET_VERSION, detail)


def _cys_pair(seq: str):
    """Positions (c1, c2) of the FR1/FR3 disulfide cysteines, or None."""
    cys = [i for i, a in enumerate(seq) if a == "C"]
    for c1 in cys:
        if c1 > FR1_CYS_MAX + 20:          # no plausible FR1 Cys even allowing a leader
            break
        for c2 in cys:
            if CYS_PAIR_MIN <= c2 - c1 <= CYS_PAIR_MAX:
                return c1, c2
    return None


def _has_j_region(seq: str) -> bool:
    return bool(_JMOTIF.search(seq[-J_CTERM_WINDOW:]) or _JTAIL.search(seq[-12:]))


def check_completeness(sequence: str) -> CompletenessResult:
    seq = sequence.strip().upper()
    n = len(seq)
    if n < MIN_DOMAIN:
        return _fail("too_short", length=n)
    if _REV_START.match(seq):
        return _fail("reversed_order")

    pair = _cys_pair(seq)
    if pair is None:
        return _fail("missing_v_anchor")
    c1, c2 = pair

    if c1 > FR1_CYS_MAX:
        return _fail("leader_or_constant_contamination", kind="leader", fr1_cys=c1)
    if n - c2 < MIN_J_TRAIL or not _has_j_region(seq):
        return _fail("missing_j_anchor", trailing=n - c2)
    if n > MAX_DOMAIN:
        return _fail("leader_or_constant_contamination", kind="constant", length=n)

    return CompletenessResult(True, None, RULESET_VERSION,
                              {"length": n, "fr1_cys": c1, "fr3_cys": c2})
