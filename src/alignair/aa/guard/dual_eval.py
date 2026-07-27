"""Stage-5 Task 5.3 — dual evaluation: from ONE ordered list of inputs, emit both a ``reference_covered``
(all profiles visible) and a ``blind`` (the fold's profiles masked) score stream.

Contract (owner-mandated):
  * Both modes consume the IDENTICAL ordered inputs (proven by a shared ``input_order_sha256``).
  * ``blind`` is the ONLY binding mode; ``reference_covered`` is diagnostic and can never feed threshold
    selection or eligibility (``binding_scores`` refuses anything else; ``assert_binding_is_blind`` guards it).
  * ``blind`` proves the COMPLETE mask path: masked HMM search + derived margins (``hmm_features(mask_groups)``),
    fusion input (``fusion.normalize`` on the blind vector), the C0/decision rules (``c0_decide``), and reason
    attribution (``assert_reason_blind`` — no masked group may surface in a reason).
  * RAW scores only — no threshold is selected or applied here. The over-length tokenizer contract is honored:
    an input past the length contract is WITHHELD (no model input, ``applicability_logit``/``energy`` = None).
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
    seq_sha256: str
    raw_length: int
    within_length_contract: bool
    applicability_logit: float | None      # learned head raw logit (None if withheld by length contract)
    energy: float | None                   # learned energy (None if withheld)
    decision_state: str                    # the guard decision state (length-withhold OR c0 rule verdict)
    decision_reason: str
    best_v_group: str
    margin: float
    masked_groups: tuple


@dataclass(frozen=True)
class ModeScores:
    mode: str                              # "reference_covered" | "blind"
    diagnostic_only: bool                  # True for reference_covered
    binding: bool                          # True only for blind
    masked_groups: tuple
    input_order_sha256: str                # sha of the ordered seq shas — proves identical ordered inputs
    scores: tuple                          # tuple[RecordScore]


@dataclass(frozen=True)
class DualEvalResult:
    fold: str
    candidate: str
    reference_covered: ModeScores
    blind: ModeScores
    identical_inputs: bool                 # ref and blind saw byte-identical ordered inputs

    def to_dict(self) -> dict:
        def _mode(m):
            d = asdict(m)
            d["scores"] = [asdict(s) for s in m.scores]
            return d
        return {"fold": self.fold, "candidate": self.candidate, "identical_inputs": self.identical_inputs,
                "reference_covered": _mode(self.reference_covered), "blind": _mode(self.blind)}


def _score_record(model, bank, fusion, tok, seq, idx, *, mask_groups, bank_checksum, device):
    """Raw scores for one input under a given mask set. Runs the FULL mask path (search→margins→fusion→C0→
    reason). Over-length inputs are withheld (no model input) per the tokenizer no-truncation contract."""
    import torch

    seq_sha = _sha(seq)
    contract = enforce_length_contract(seq)
    feats = hmm_features(bank, seq, mask_groups=mask_groups)          # masked HMM search + derived margins
    if not contract.ok:
        verdict = contract.verdict                                    # withheld: insufficient_evidence
        assert_reason_blind(verdict.reason, mask_groups)              # (length reason names no group; safe)
        return RecordScore(idx, seq_sha, contract.raw_length, False, None, None,
                           verdict.state, verdict.reason, feats.best_v_group, round(feats.margin, 6),
                           tuple(sorted(mask_groups)))
    verdict = c0_decide(feats)                                        # C0/decision rules on the (blind) features
    assert_reason_blind(verdict.reason, mask_groups)                 # reason attribution must be blind
    vec = feats.to_vector().tolist()
    hmm_norm = fusion.normalize(vec, feature_names=FEATURE_NAMES, bank_checksum=bank_checksum)  # fusion input
    ids, mask = tok.encode(seq)
    model.eval()
    with torch.no_grad():
        out = model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device),
                    hmm_norm=hmm_norm.unsqueeze(0).to(device) if model.cfg["fusion"] else None)
    return RecordScore(idx, seq_sha, contract.raw_length, True,
                       float(out["applicability"].squeeze().item()), float(out["energy"].squeeze().item()),
                       verdict.state, verdict.reason, feats.best_v_group, round(feats.margin, 6),
                       tuple(sorted(mask_groups)))


def _mode_scores(model, bank, fusion, tok, seqs, *, mode, mask_groups, bank_checksum, device):
    scores = tuple(_score_record(model, bank, fusion, tok, s, i, mask_groups=mask_groups,
                                 bank_checksum=bank_checksum, device=device) for i, s in enumerate(seqs))
    order = _sha(json.dumps([sc.seq_sha256 for sc in scores]))
    return ModeScores(mode=mode, diagnostic_only=(mode == "reference_covered"), binding=(mode == "blind"),
                      masked_groups=tuple(sorted(mask_groups)), input_order_sha256=order, scores=scores)


def dual_eval(model, bank, fusion, tok, records, *, fold, candidate, bank_checksum, device="cpu") -> DualEvalResult:
    """Score the SAME ordered `records` twice: reference-covered (no mask) and blind (fold profiles masked)."""
    seqs = [r["aa"] for r in records]                                # single ordered input list, used by both
    masked = masked_profiles_for(fold)
    ref = _mode_scores(model, bank, fusion, tok, seqs, mode="reference_covered",
                       mask_groups=frozenset(), bank_checksum=bank_checksum, device=device)
    blind = _mode_scores(model, bank, fusion, tok, seqs, mode="blind",
                         mask_groups=masked, bank_checksum=bank_checksum, device=device)
    return DualEvalResult(fold=fold, candidate=candidate, reference_covered=ref, blind=blind,
                          identical_inputs=(ref.input_order_sha256 == blind.input_order_sha256))


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
