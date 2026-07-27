"""Stage-5 Task 5.3 — MOCK one-shot on SYNTHETIC sealed fixtures.

Exercises the complete threshold-freeze -> challenge-unlock -> dual-evaluation sequence and hard-asserts every
input/model/mask/threshold hash and each of the five mechanical access-barrier invariants. NOTHING here reads a
real fresh-selection or loco-challenge file, and nothing produced here is a production operating point.

On success it then writes the REAL pre-unlock frozen manifest (every hash frozen BEFORE the first fresh read,
built only from already-committed burned-dev artifacts + the LOCO checkpoints) and STOPS.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/mock_oneshot.py
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import torch

from alignair.aa.guard.access import (
    AccessViolation, ChallengeLoader, FOLD_GATE, HashMismatch, PREDECLARED_GRID, SCORE_DIRECTIONS,
    SelectionLoader, ThresholdArtifactMissing, freeze_fold_threshold, threshold_grid_hash,
    verify_challenge_access,
)
from alignair.aa.guard.checkpoint import CheckpointProvenance
from alignair.aa.guard.classes import CLASS_MAP_HASH
from alignair.aa.guard.dual_eval import assert_binding_is_blind, binding_scores, dual_eval
from alignair.aa.guard.features import FEATURE_NAMES, hmm_features
from alignair.aa.guard.freeze import (
    feature_order_hash, freeze_pre_unlock_bundle, preprocessing_hash,
)
from alignair.aa.guard.fusion import fit_fusion
from alignair.aa.guard.loco import LOCO_FOLDS, LOCO_MASK_HASH, masked_profiles_for
from alignair.aa.guard.model import GuardModel
from alignair.aa.guard.profiles import build_bank_fixture
from alignair.aa.guard.tokenizer import AaTokenizer

ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability")
SCRATCH = Path("/tmp/claude-1000/-home-thomas-Desktop-AlignAIR/376842b6-033f-49b5-a34c-02e6621fd496/"
               "scratchpad/mock_oneshot")
LOCO_MAN = json.load(open(ART / "loco_run_manifest.json"))
STAGE4_MAN = json.load(open(ART / "stage4_run_manifest.json"))

# ---- synthetic sealed fixtures (fabricated sequences; never a real cohort) ----------------------------
IGH_V = b">h1\nQVQLVQSGAEVKKPGASVKVSCKASGYTF\n>h2\nQVQLVQSGAEVKKPGSSVKVSCKASGGTF\n>h3\nEVQLVESGGGLVQPGGSLRLSCAASGFTF\n"
IGH_J = b">hj1\nWGQGTLVTVSSASTKG\n>hj2\nWGQGTTVTVSSASTKG\n>hj3\nWGQGTMVTVSSASTKG\n"
IGK_V = b">k1\nDIQMTQSPSSLSASVGDRVTITCRASQSI\n>k2\nDIQMTQSPSSLSASVGDRVTITCRASQGI\n>k3\nDIQLTQSPSFLSASVGDRVTITCRASQGI\n"
IGK_J = b">kj1\nFGQGTKVEIKRTVA\n>kj2\nFGQGTKLEIKRTVA\n>kj3\nFGGGTKVEIKRTVA\n"
MOCK_FOLD = "igk_igl"
SYNTH_SELECTION = [{"aa": "QVQLVQSGAEVKKPGASVKVSCKASGYTFAAAAWGQGTLVTVSSASTKG", "expected_state": "supported"},
                   {"aa": "DIQMTQSPSSLSASVGDRVTITCRASQSIAAAAFGQGTKVEIKRTVA", "expected_state": "unknown_ood"}]
SYNTH_CHALLENGE = [{"aa": "DIQLTQSPSFLSASVGDRVTITCRASQGIAAAAFGGGTKVEIKRTVA", "expected_state": "unknown_ood"},
                   {"aa": "EVQLVESGGGLVQPGGSLRLSCAASGFTFAAAAWGQGTMVTVSSASTKG", "expected_state": "supported"}]


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _state_hash(model) -> str:
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    return _sha_bytes(buf.getvalue())


def _norm_hash(fusion) -> str:
    return _sha_bytes(fusion.mean.tobytes() + fusion.std.tobytes())


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _check(checks, name, fn):
    """Run a hard assertion; record pass, re-raise on failure (a failed mock invalidates the whole gate)."""
    fn()
    checks.append(name)
    print(f"  [ok] {name}", flush=True)


def run_mock() -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    checks: list = []

    # 1. synthetic model / bank / fusion / tokenizer
    bank = build_bank_fixture(regions={"human_igh": {"v": IGH_V, "j": IGH_J},
                                        "human_igl": {"v": IGK_V, "j": IGK_J}})
    ck = bank.checksum()
    tok = AaTokenizer()
    vecs = [hmm_features(bank, r["aa"]).to_vector().tolist() for r in SYNTH_SELECTION + SYNTH_CHALLENGE]
    fusion = fit_fusion(FEATURE_NAMES, vecs, ck)
    model = GuardModel("C2", len(FEATURE_NAMES), width=16, depth=1, heads=2)

    # 2. write synthetic sealed fixtures into role-scoped roots
    sel_root, chal_root = SCRATCH / "selection", SCRATCH / "challenge"
    sel_root.mkdir(parents=True, exist_ok=True); chal_root.mkdir(parents=True, exist_ok=True)
    (sel_root / "selection.jsonl").write_text("\n".join(json.dumps(r) for r in SYNTH_SELECTION) + "\n")
    (chal_root / "challenge.jsonl").write_text("\n".join(json.dumps(r) for r in SYNTH_CHALLENGE) + "\n")

    # 3. freeze the pre-unlock bundle (synthetic model, real global identities)
    bundle = freeze_pre_unlock_bundle(
        model_hash=_state_hash(model), normalization_hash=_norm_hash(fusion),
        source_vocab_hash=hashlib.sha256(json.dumps(["cat_a", "cat_b"]).encode()).hexdigest(),
        bank_checksum=ck, evaluator_commit=_git_head(),
        cohort_manifest_hash=_sha_bytes((ART / "FROZEN_COHORT_MANIFEST.json").read_bytes()),
        threshold_grid_hash=threshold_grid_hash())
    sel_man_hash = _sha_bytes((sel_root / "selection.jsonl").read_bytes())

    # 4. BARRIER 1 — the fresh-selection loader cannot reach a challenge file
    sel_loader = SelectionLoader(sel_root, chal_root)
    def _b1():
        assert sel_loader.load(sel_root / "selection.jsonl")           # selection ok
        try:
            sel_loader.load(chal_root / "challenge.jsonl"); raise AssertionError("should have refused")
        except AccessViolation:
            pass
    _check(checks, "barrier1_selection_cannot_open_challenge", _b1)

    # 5. dual-eval on the SELECTION fixtures (via the selection loader) -> blind binds, ref is diagnostic
    sel_recs = [json.loads(x) for x in sel_loader.load(sel_root / "selection.jsonl").decode().splitlines() if x.strip()]
    sel_res = dual_eval(model, bank, fusion, tok, sel_recs, fold=MOCK_FOLD, candidate="C2", bank_checksum=ck)
    def _b5():
        assert sel_res.identical_inputs
        assert binding_scores(sel_res) is sel_res.blind
        try:
            assert_binding_is_blind(sel_res.reference_covered); raise AssertionError("ref bound")
        except ValueError:
            pass
    _check(checks, "dual_eval_selection_binding_is_blind_only", _b5)

    # 6. threshold-freeze on the BLIND selection scores (synthetic zero-error guardband counts; mock=True)
    thr_path = SCRATCH / "fold_threshold.json"
    guardband = [(PREDECLARED_GRID[2], 0, 74, 0, 59)]                   # (0/74) passes 5%-gate FAR, (0/59) FRR
    artifact = freeze_fold_threshold(MOCK_FOLD, sel_res.blind, fold_checkpoint_sha256=_state_hash(model),
                                     selection_manifest_sha256=sel_man_hash, frozen_bundle=bundle,
                                     guardband_grid=guardband, out_path=thr_path, mock=True,
                                     marker="mock_oneshot")
    def _b6():
        assert thr_path.exists() and artifact.is_complete()
        assert artifact.mask_hash == LOCO_MASK_HASH and artifact.grid_digest() == bundle.hashes["threshold_grid"]
    _check(checks, "threshold_freeze_writes_complete_hash_bound_artifact", _b6)

    excluded = (MOCK_FOLD, tuple(sorted(masked_profiles_for(MOCK_FOLD))))

    # 7. BARRIER 2 — challenge refused with no artifact present
    def _b2():
        loader = ChallengeLoader(challenge_root=chal_root, selection_root=sel_root,
                                 threshold_artifact_path=SCRATCH / "does_not_exist.json",
                                 audit_path=SCRATCH / "audit.jsonl", frozen_bundle=bundle,
                                 expected_selection_manifest_sha256=sel_man_hash, expected_excluded=excluded)
        try:
            loader.open(chal_root / "challenge.jsonl"); raise AssertionError("opened without artifact")
        except ThresholdArtifactMissing:
            pass
    _check(checks, "barrier2_challenge_refused_without_artifact", _b2)

    # 8. BARRIER 5 — a changed hash hard-fails, no silent recompute
    def _b5b():
        tampered = freeze_pre_unlock_bundle(
            model_hash="Z" * 64, normalization_hash=_norm_hash(fusion),
            source_vocab_hash=bundle.hashes["source_vocab"], bank_checksum=ck, evaluator_commit=_git_head(),
            cohort_manifest_hash=bundle.hashes["cohort_manifest"], threshold_grid_hash=threshold_grid_hash())
        try:
            verify_challenge_access(artifact, tampered, expected_selection_manifest_sha256=sel_man_hash,
                                    expected_excluded=excluded); raise AssertionError("drift not caught")
        except HashMismatch:
            pass
    _check(checks, "barrier5_changed_hash_hard_fails", _b5b)

    # 9. BARRIER 4 — challenge unlock writes an atomic audit record BEFORE returning any challenge bytes
    audit_path = SCRATCH / "audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    chal_loader = ChallengeLoader(challenge_root=chal_root, selection_root=sel_root,
                                  threshold_artifact_path=thr_path, audit_path=audit_path, frozen_bundle=bundle,
                                  expected_selection_manifest_sha256=sel_man_hash, expected_excluded=excluded,
                                  marker="mock_oneshot")
    chal_bytes = chal_loader.open(chal_root / "challenge.jsonl")
    def _b4():
        assert audit_path.exists(), "audit not written"
        rec = [json.loads(x) for x in audit_path.read_text().splitlines() if x.strip()][-1]
        assert rec["event"] == "challenge_unlock" and rec["fold"] == MOCK_FOLD
        assert rec["bundle_sha256"] == bundle.bundle_sha256()
    _check(checks, "barrier4_atomic_audit_before_scoring", _b4)

    # 10. dual-EVALUATION on the challenge fixtures -> RAW scores only (no threshold applied here)
    chal_recs = [json.loads(x) for x in chal_bytes.decode().splitlines() if x.strip()]
    chal_res = dual_eval(model, bank, fusion, tok, chal_recs, fold=MOCK_FOLD, candidate="C2", bank_checksum=ck)
    def _b10():
        assert chal_res.identical_inputs
        assert chal_res.blind.masked_groups == tuple(sorted(masked_profiles_for(MOCK_FOLD)))
        for sc in chal_res.blind.scores:                              # blind reasons never name a masked group
            assert "human_igk" not in sc.decision_reason and "human_igl" not in sc.decision_reason
            assert sc.best_v_group not in ("human_igk", "human_igl")
    _check(checks, "dual_eval_challenge_raw_scores_blind_masked", _b10)

    # 11. full hash-set consistency
    def _b11():
        from alignair.aa.guard.freeze import REQUIRED_SLOTS
        assert set(bundle.hashes) == set(REQUIRED_SLOTS)
        assert artifact.bundle_sha256 == bundle.bundle_sha256()
        assert bundle.hashes["mask_graph"] == LOCO_MASK_HASH
        assert bundle.hashes["class_map"] == CLASS_MAP_HASH
        assert bundle.hashes["feature_order"] == feature_order_hash()
        assert bundle.hashes["preprocessing"] == preprocessing_hash()
    _check(checks, "full_hash_set_consistent", _b11)

    return {"passed": True, "checks": checks, "n_checks": len(checks),
            "bundle_sha256": bundle.bundle_sha256(), "artifact_digest": artifact.digest(),
            "mock_fold": MOCK_FOLD, "n_selection": len(SYNTH_SELECTION), "n_challenge": len(SYNTH_CHALLENGE),
            "raw_score_sample": {"reference_covered": [s.applicability_logit for s in chal_res.reference_covered.scores],
                                 "blind": [s.applicability_logit for s in chal_res.blind.scores]}}


def _model_digests() -> dict:
    """Read each LOCO checkpoint's bound provenance digest + normalization hash (burned-dev; not fresh data)."""
    out = {}
    for p in sorted(glob.glob(str(LOCO_MAN["checkpoint_dir"]) + "/*.final.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        prov, be = ck["provenance"], ck["bound_extra"]
        cp = CheckpointProvenance(tokenizer=prov["tokenizer"], bank_checksum=prov["bank_checksum"],
                                  feature_order=tuple(prov["feature_order"]), class_map_hash=prov["class_map_hash"],
                                  loss_mask_config=prov["loss_mask_config"],
                                  training_manifest_hash=prov["training_manifest_hash"],
                                  partition_hash=prov["partition_hash"], seed=prov["seed"],
                                  software_versions=prov["software_versions"])
        tag = Path(p).stem.replace(".final", "")                       # e.g. C1.s2.igk_igl
        cid, _seed, fold = tag.split(".")
        out[f"{cid}.{fold}"] = {"candidate": cid, "fold": fold, "checkpoint_provenance_digest": cp.digest(),
                                "normalization_stats_sha256": be["normalization_stats_sha256"],
                                "partition_hash": be["partition_hash"]}
    return out


def write_pre_unlock_manifest() -> dict:
    """Every hash frozen BEFORE the first fresh read (globals + per-fold + per-model), plus the per-model
    frozen-bundle sha the challenge loader will verify against. Built only from committed/burned artifacts."""
    per_model = _model_digests()
    glob_hashes = {
        "class_map": CLASS_MAP_HASH, "profile_bank": LOCO_MAN["bank_checksum"],
        "feature_order": feature_order_hash(), "mask_graph": LOCO_MASK_HASH,
        "preprocessing": preprocessing_hash(), "threshold_grid": threshold_grid_hash(),
        "evaluator_commit": _git_head(),
        "cohort_manifest": _sha_bytes((ART / "FROZEN_COHORT_MANIFEST.json").read_bytes()),
        "loco_run_manifest": LOCO_MAN["manifest_sha256"], "stage4_run_manifest": STAGE4_MAN["manifest_sha256"],
    }
    per_fold = {g: {"source_vocab": LOCO_MAN["folds"][g]["source_vocab_sha256"],
                    "n_sources": LOCO_MAN["folds"][g]["n_sources"], "gate_kind": FOLD_GATE[g],
                    "excluded_group": g, "masked_profiles": sorted(masked_profiles_for(g))}
                for g in LOCO_FOLDS}
    # per-model frozen bundle sha (what a fold-threshold artifact must match at unlock)
    bundles = {}
    for key, m in per_model.items():
        b = freeze_pre_unlock_bundle(
            model_hash=m["checkpoint_provenance_digest"], normalization_hash=m["normalization_stats_sha256"],
            source_vocab_hash=per_fold[m["fold"]]["source_vocab"], bank_checksum=glob_hashes["profile_bank"],
            evaluator_commit=glob_hashes["evaluator_commit"], cohort_manifest_hash=glob_hashes["cohort_manifest"],
            threshold_grid_hash=glob_hashes["threshold_grid"])
        bundles[key] = b.bundle_sha256()
    manifest = {
        "artifact": "pre_unlock_frozen_manifest", "immutable": True,
        "purpose": "Every hash frozen BEFORE the first fresh-selection read. The fold-threshold artifact "
                   "produced on fresh selection, and every challenge unlock, is verified against these.",
        "score_directions": SCORE_DIRECTIONS, "predeclared_grid": [dict(g) for g in PREDECLARED_GRID],
        "fold_gate": FOLD_GATE, "global": glob_hashes, "per_fold": per_fold, "per_model": per_model,
        "per_model_bundle_sha256": bundles,
        "not_opened": ["fresh_selection", "loco_challenge", "far_ood", "sealed locked_test.jsonl"],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (ART / "PRE_UNLOCK_FROZEN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main():
    print("[mock] running synthetic one-shot ...", flush=True)
    result = run_mock()
    (ART / "MOCK_ONESHOT_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[mock] PASS — {result['n_checks']}/{result['n_checks']} checks OK", flush=True)
    man = write_pre_unlock_manifest()
    print(f"[mock] pre-unlock manifest frozen: {man['manifest_sha256'][:16]} "
          f"({len(man['per_model'])} models, {len(man['per_fold'])} folds)", flush=True)
    print("[mock] STOP — mock passed + pre-unlock manifest frozen. NO fresh selection / loco_challenge opened.",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
