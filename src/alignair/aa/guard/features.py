"""Task 2.3 — competitive profile-HMM search → fixed-length HMM feature vector.

Each query sequence is searched against every profile in the bank (deterministic ``pyhmmer.plan7.Pipeline``
with ``bias_filter=False``). We assemble PAIRED V/J evidence for the winning locus plus the competitive
margin of human-IGH against the best alternative locus, and emit a fixed-length numeric vector (with
``FEATURE_NAMES``) alongside string metadata (winning/alternative group) the C0 rules consume.

A complete domain requires BOTH a V and a J hit in N→C order with sufficient combined query coverage; a
V-only hit never sets ``coverage_ok``/``order_ok``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_NEG = -1000.0                      # finite sentinel for "no hit" scores in the vector
_BIG_E = 1e9                        # sentinel E-value for "no hit"


def _search_one(hmm, seq_block, abc):
    """Best-domain evidence for one profile against one query; None if no hit."""
    from pyhmmer import plan7
    hits = plan7.Pipeline(abc, bias_filter=False).search_hmm(hmm, seq_block)
    if not len(hits):
        return None
    hit = max(hits, key=lambda h: h.score)
    dom = hit.best_domain
    a = dom.alignment
    return {"score": float(dom.score), "evalue": float(hit.evalue),
            "q_from": int(a.target_from), "q_to": int(a.target_to),
            "p_from": int(a.hmm_from), "p_to": int(a.hmm_to), "p_len": int(a.hmm_length)}


def _q_score(res):
    return res["score"] if res else _NEG


# Fixed numeric feature order (to_vector); string group metadata is carried on HmmFeatures, not the vector.
FEATURE_NAMES = [
    "human_igh_score", "best_alt_score", "margin",
    "best_v_score", "best_v_evalue_log10", "best_j_score", "best_j_evalue_log10",
    "v_qfrac", "v_pfrac", "j_qfrac", "j_pfrac",
    "v_before_j", "inter_vj_gap", "combined_coverage", "unaligned_n", "unaligned_c",
    "n_significant_hits", "has_v", "has_j", "order_ok", "coverage_ok", "seq_len",
]


@dataclass
class HmmFeatures:
    best_v_group: str
    best_v_species: str
    best_j_group: str
    best_alt_group: str
    human_igh_score: float
    best_alt_score: float
    margin: float
    best_v_score: float
    best_v_evalue: float
    best_j_score: float
    best_j_evalue: float
    v_qfrac: float
    v_pfrac: float
    j_qfrac: float
    j_pfrac: float
    v_before_j: bool
    inter_vj_gap: int
    combined_coverage: float
    unaligned_n: int
    unaligned_c: int
    n_significant_hits: int
    has_v: bool
    has_j: bool
    order_ok: bool
    coverage_ok: bool
    seq_len: int
    per_group_v_score: dict = field(default_factory=dict)
    masked_groups: tuple = ()          # blind-LOCO: profile groups excluded from the competitive set

    def to_vector(self) -> np.ndarray:
        def _l10(e):
            return float(np.log10(e)) if e and e < _BIG_E else 9.0
        vec = [
            self.human_igh_score, self.best_alt_score, self.margin,
            self.best_v_score, _l10(self.best_v_evalue), self.best_j_score, _l10(self.best_j_evalue),
            self.v_qfrac, self.v_pfrac, self.j_qfrac, self.j_pfrac,
            float(self.v_before_j), float(self.inter_vj_gap), self.combined_coverage,
            float(self.unaligned_n), float(self.unaligned_c), float(self.n_significant_hits),
            float(self.has_v), float(self.has_j), float(self.order_ok), float(self.coverage_ok),
            float(self.seq_len),
        ]
        return np.asarray(vec, dtype=np.float64)


def _species_of(group: str) -> str:
    if group.startswith("human_") or group in ("tra", "trb", "trg", "trd"):
        return "human"
    if group == "murine_vh":
        return "mouse"
    if group == "camelid_igh":
        return "camelid"
    return "other"


def hmm_features(bank, seq: str, *, evalue_sig: float = 1e-3, coverage_min: float = 0.80,
                 mask_groups=frozenset()) -> HmmFeatures:
    """Competitive search of `seq` against every profile in `bank` → fixed feature record.

    Blind-LOCO: `mask_groups` are removed from the competitive set BEFORE any scoring, so the masked
    profiles influence neither the raw features, the competitive margins, nor any derived value. The masked
    groups therefore can never be the winner/best-alternative and can never surface in a decision reason."""
    from pyhmmer import easel

    from alignair.aa.guard.profiles import _amino
    mask_groups = frozenset(mask_groups)
    abc = _amino()             # SAME shared alphabet the bank's HMMs were built with (avoids ProfileConfig)
    q = easel.TextSequence(name=b"query", sequence=seq).digitize(abc)
    block = easel.DigitalSequenceBlock(abc, [q])
    L = len(seq)

    # search every profile EXCEPT masked groups; collect per group/region
    res = {}  # (group, region) -> hit dict or None
    for group, regions in bank.profiles.items():
        if group in mask_groups:
            continue           # blind: excluded from the competitive search entirely
        for region, (hmm, _sha) in regions.items():
            res[(group, region)] = _search_one(hmm, block, abc)

    def v_res(group):
        return res.get((group, "v")) or res.get((group, "domain"))

    groups = [g for g in bank.profiles if g not in mask_groups]     # masked groups fully excluded
    per_group_v = {g: _q_score(v_res(g)) for g in groups}
    human_igh_score = per_group_v.get("human_igh", _NEG)
    alts = {g: s for g, s in per_group_v.items() if g != "human_igh"}
    best_alt_group = max(alts, key=alts.get) if alts else ""
    best_alt_score = alts.get(best_alt_group, _NEG)
    margin = human_igh_score - best_alt_score

    best_v_group = max(per_group_v, key=per_group_v.get) if per_group_v else ""
    best_j_group, best_j_score, best_j_ev = "", _NEG, _BIG_E
    for g in groups:
        r = res.get((g, "j"))
        if r and r["score"] > best_j_score:
            best_j_group, best_j_score, best_j_ev = g, r["score"], r["evalue"]

    # domain geometry from the WINNING V group's V and its own J
    win_v = v_res(best_v_group)
    win_j = res.get((best_v_group, "j"))
    has_v = win_v is not None
    has_j = win_j is not None
    best_v_score = win_v["score"] if has_v else _NEG
    best_v_ev = win_v["evalue"] if has_v else _BIG_E

    v_qfrac = (win_v["q_to"] - win_v["q_from"] + 1) / L if has_v else 0.0
    v_pfrac = (win_v["p_to"] - win_v["p_from"] + 1) / win_v["p_len"] if has_v else 0.0
    j_qfrac = (win_j["q_to"] - win_j["q_from"] + 1) / L if has_j else 0.0
    j_pfrac = (win_j["p_to"] - win_j["p_from"] + 1) / win_j["p_len"] if has_j else 0.0

    v_before_j = bool(has_v and has_j and win_v["q_to"] < win_j["q_from"])
    inter_vj_gap = (win_j["q_from"] - win_v["q_to"] - 1) if (has_v and has_j) else 0
    if has_v and has_j:
        lo, hi = min(win_v["q_from"], win_j["q_from"]), max(win_v["q_to"], win_j["q_to"])
        combined_coverage = (hi - lo + 1) / L
        unaligned_n = win_v["q_from"] - 1
        unaligned_c = L - win_j["q_to"]
    elif has_v:
        combined_coverage = v_qfrac
        unaligned_n = win_v["q_from"] - 1
        unaligned_c = L - win_v["q_to"]
    else:
        combined_coverage = unaligned_n = unaligned_c = 0.0

    n_sig = sum(1 for r in res.values() if r and r["evalue"] < evalue_sig)
    order_ok = v_before_j
    coverage_ok = bool(has_v and has_j and combined_coverage >= coverage_min)

    return HmmFeatures(
        best_v_group=best_v_group, best_v_species=_species_of(best_v_group),
        best_j_group=best_j_group, best_alt_group=best_alt_group,
        human_igh_score=human_igh_score, best_alt_score=best_alt_score, margin=margin,
        best_v_score=best_v_score, best_v_evalue=best_v_ev,
        best_j_score=best_j_score, best_j_evalue=best_j_ev,
        v_qfrac=v_qfrac, v_pfrac=v_pfrac, j_qfrac=j_qfrac, j_pfrac=j_pfrac,
        v_before_j=v_before_j, inter_vj_gap=int(inter_vj_gap), combined_coverage=combined_coverage,
        unaligned_n=int(unaligned_n), unaligned_c=int(unaligned_c), n_significant_hits=n_sig,
        has_v=has_v, has_j=has_j, order_ok=order_ok, coverage_ok=coverage_ok, seq_len=L,
        per_group_v_score=per_group_v, masked_groups=tuple(sorted(mask_groups)))
