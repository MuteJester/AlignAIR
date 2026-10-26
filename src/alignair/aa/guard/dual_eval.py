"""Stage-5 Task 5.3 — dual evaluation: from ONE ordered list of inputs, emit both a ``reference_covered``
(all profiles visible) and a ``blind`` (the fold's profiles masked) score stream.

Contract (owner-mandated):
  * Both modes consume the IDENTICAL ordered inputs — proven by a shared ``input_order_sha256`` (sequence
    identities) AND ``input_contract_digest`` (the ordered independence-unit + label contract).
  * ``blind`` is the ONLY binding mode; ``reference_covered`` is diagnostic and can never feed threshold
    selection or eligibility (``binding_scores`` refuses anything else; ``assert_binding_is_blind`` guards it).
  * ``blind`` proves the COMPLETE mask path: masked HMM search + derived margins (``hmm_features(mask_groups)``),
    fusion input (``fusion.normalize`` on the blind vector), the C0/decision rules (``c0_decide``), and reason
    attribution (``assert_reason_blind`` — no masked group may surface in a reason).
  * RAW outputs only — every decision-relevant signal is preserved (5 completeness logits, 8 domain-type
    logits, applicability logit, energy, the structural prefilter, the decision-relevant HMM evidence, the
    expected label, and the independence-unit identity) so the frozen decision is fully reconstructable WITHOUT
    rerunning inference. No threshold is selected or applied here.
  * The over-length tokenizer contract is honored: an input past the length contract is WITHHELD (no model
    input, learned outputs = None).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from alignair.aa.guard.c0_rules import c0_decide
from alignair.aa.guard.features import FEATURE_NAMES, hmm_features
from alignair.aa.guard.loco import assert_reason_blind, masked_profiles_for
from alignair.aa.guard.tokenizer import enforce_length_contract


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass(frozen=True)
class RecordScore:
    idx: int
    record_id_sha256: str                  # hashed record ID (independence metadata; never the raw id)
    seq_sha256: str
    raw_length: int
    within_length_contract: bool
    structural_prefilter: dict             # {"length_contract": ok/over_length, "geometry": complete/incomplete}
    applicability_logit: float | None      # learned binary head (None if withheld)
    energy: float | None                   # open-set free energy = -logsumexp(type) (None if withheld)
    completeness_logits: tuple | None      # 5 raw logits (None if withheld)
    domain_type_logits: tuple | None       # 8 raw logits (None if withheld)
    hmm_evidence: dict                     # decision-relevant HMM fields (geometry, winner, margins, scores)
    c0_state: str
    c0_reason: str
    expected_state: str                    # frozen label
    expected_category: str                 # frozen category
    group: str                             # frozen cohort/selection group
    unit: dict                             # independence identity: lineage/parent_lineage/donor/study/role
    masked_groups: tuple

    def contract_tuple(self) -> tuple:
        """The ordered per-record independence + label contract (bound by ModeScores.input_contract_digest)."""
        u = self.unit or {}
        return (self.record_id_sha256, self.group, self.expected_category, self.expected_state,
                u.get("lineage"), u.get("parent_lineage"), u.get("donor"), u.get("study"), u.get("role"))


@dataclass(frozen=True)
class ModeScores:
    mode: str                              # "reference_covered" | "blind"
    diagnostic_only: bool                  # True for reference_covered
    binding: bool                          # True only for blind
    masked_groups: tuple
    input_order_sha256: str                # sha of the ordered seq shas
    input_contract_digest: str             # sha of the ordered independence-unit + label contract
    scores: tuple                          # tuple[RecordScore]


@dataclass(frozen=True)
class DualEvalResult:
    fold: str
    candidate: str
    reference_covered: ModeScores
    blind: ModeScores
    identical_inputs: bool                 # ref and blind saw byte-identical ordered inputs AND contract

    def to_dict(self) -> dict:
        def _mode(m):
            d = asdict(m)
            d["scores"] = [asdict(s) for s in m.scores]
            return d
        return {"fold": self.fold, "candidate": self.candidate, "identical_inputs": self.identical_inputs,
                "reference_covered": _mode(self.reference_covered), "blind": _mode(self.blind)}


def _unit_of(rec: dict) -> dict:
    return {"lineage": rec.get("lineage_key") or rec.get("lineage"),
            "parent_lineage": rec.get("parent_lineage") or rec.get("lineage_key") or rec.get("lineage"),
            "donor": rec.get("donor") or rec.get("subject"), "study": rec.get("study"),
            "role": rec.get("role_split") or rec.get("role")}


def _hmm_evidence(f) -> dict:
    return {"has_v": bool(f.has_v), "has_j": bool(f.has_j), "order_ok": bool(f.order_ok),
            "coverage_ok": bool(f.coverage_ok), "combined_coverage": round(float(f.combined_coverage), 6),
            "best_v_group": f.best_v_group, "best_alt_group": f.best_alt_group, "best_j_group": f.best_j_group,
            "human_igh_score": round(float(f.human_igh_score), 6), "best_alt_score": round(float(f.best_alt_score), 6),
            "margin": round(float(f.margin), 6), "best_v_score": round(float(f.best_v_score), 6),
            "best_j_score": round(float(f.best_j_score), 6), "n_significant_hits": int(f.n_significant_hits)}


def _score_record(model, bank, fusion, tok, rec, idx, *, mask_groups, bank_checksum, device):
    """Raw outputs for one record under a given mask set. Runs the FULL mask path (search→margins→fusion→C0→
    reason). Over-length inputs are withheld (no model input) per the tokenizer no-truncation contract."""
    import torch

    seq = rec["aa"]
    rid = _sha(str(rec.get("id", seq)))
    feats = hmm_features(bank, seq, mask_groups=mask_groups)          # masked HMM search + derived margins
    contract = enforce_length_contract(seq)
    geometry = "complete" if (feats.has_v and feats.has_j and feats.order_ok and feats.coverage_ok) else "incomplete"
    prefilter = {"length_contract": "ok" if contract.ok else "over_length", "geometry": geometry}
    base = dict(idx=idx, record_id_sha256=rid, seq_sha256=_sha(seq), raw_length=contract.raw_length,
                structural_prefilter=prefilter, hmm_evidence=_hmm_evidence(feats),
                expected_state=rec.get("expected_state", ""), expected_category=rec.get("category", ""),
                group=rec.get("group", ""), unit=_unit_of(rec), masked_groups=tuple(sorted(mask_groups)))
    if not contract.ok:
        v = contract.verdict                                          # withheld: insufficient_evidence
        assert_reason_blind(v.reason, mask_groups)                    # (length reason names no group; safe)
        return RecordScore(within_length_contract=False, applicability_logit=None, energy=None,
                           completeness_logits=None, domain_type_logits=None, c0_state=v.state,
                           c0_reason=v.reason, **base)
    verdict = c0_decide(feats)                                        # C0/decision rules on the (blind) features
    assert_reason_blind(verdict.reason, mask_groups)                 # reason attribution must be blind
    vec = feats.to_vector().tolist()
    hmm_norm = fusion.normalize(vec, feature_names=FEATURE_NAMES, bank_checksum=bank_checksum)  # fusion input
    ids, mask = tok.encode(seq)
    model.eval()
    with torch.no_grad():
        out = model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device),
                    hmm_norm=hmm_norm.unsqueeze(0).to(device) if model.cfg["fusion"] else None)
    return RecordScore(
        within_length_contract=True,
        applicability_logit=float(out["applicability"].squeeze().item()),
        energy=float(out["energy"].squeeze().item()),
        completeness_logits=tuple(round(x, 6) for x in out["completeness"].squeeze(0).tolist()),
        domain_type_logits=tuple(round(x, 6) for x in out["domain_type"].squeeze(0).tolist()),
        c0_state=verdict.state, c0_reason=verdict.reason, **base)


def _mode_scores(model, bank, fusion, tok, records, *, mode, mask_groups, bank_checksum, device):
    scores = tuple(_score_record(model, bank, fusion, tok, r, i, mask_groups=mask_groups,
                                 bank_checksum=bank_checksum, device=device) for i, r in enumerate(records))
    order = _sha(json.dumps([sc.seq_sha256 for sc in scores]))
    contract = _sha(json.dumps([list(sc.contract_tuple()) for sc in scores]))
    return ModeScores(mode=mode, diagnostic_only=(mode == "reference_covered"), binding=(mode == "blind"),
                      masked_groups=tuple(sorted(mask_groups)), input_order_sha256=order,
                      input_contract_digest=contract, scores=scores)


def dual_eval(model, bank, fusion, tok, records, *, fold, candidate, bank_checksum, device="cpu") -> DualEvalResult:
    """Score the SAME ordered `records` twice: reference-covered (no mask) and blind (fold profiles masked)."""
    masked = masked_profiles_for(fold)
    ref = _mode_scores(model, bank, fusion, tok, records, mode="reference_covered",
                       mask_groups=frozenset(), bank_checksum=bank_checksum, device=device)
    blind = _mode_scores(model, bank, fusion, tok, records, mode="blind",
                         mask_groups=masked, bank_checksum=bank_checksum, device=device)
    identical = (ref.input_order_sha256 == blind.input_order_sha256
                 and ref.input_contract_digest == blind.input_contract_digest)
    return DualEvalResult(fold=fold, candidate=candidate, reference_covered=ref, blind=blind,
                          identical_inputs=identical)


def assert_binding_is_blind(mode: ModeScores) -> None:
    """Structural guard: only the blind mode may bind thresholds/eligibility. Raises on reference_covered."""
    if mode.mode != "blind" or not mode.binding or mode.diagnostic_only:
        raise ValueError(f"binding refused: mode {mode.mode!r} is diagnostic, only 'blind' may bind eligibility")


def binding_scores(result: DualEvalResult) -> ModeScores:
    """Eligibility/thresholds consume ONLY blind scores; reference_covered is structurally refused."""
    if result.reference_covered.binding:
        raise ValueError("reference_covered must never be binding")
    assert_binding_is_blind(result.blind)
    return result.blind
