"""Stage-5 Task 5.3 — MOCK one-shot on SYNTHETIC sealed fixtures (corrected scientific contract).

Exercises the complete threshold-freeze -> challenge-unlock -> dual-evaluation sequence with ALL per-group
failure counts DERIVED INTERNALLY from synthetic predictions + labels (no injected counts), and hard-asserts
every input/model/mask/threshold hash plus each access-barrier invariant. NOTHING here reads a real fresh
selection or loco-challenge file, and nothing produced here is a production operating point.

Demonstrations (all from derived counts): a passing multi-group threshold; a point failing ONE group despite
aggregate/pooled success (no pooling); parent-lineage aggregation; real vs simulator FRR separation; hash-drift
rejection; and a no-admissible-threshold STOP.

On success it writes the REAL pre-unlock frozen manifest (built only from committed/burned artifacts + the LOCO
checkpoints, binding actual checkpoint-file bytes) and STOPS.

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
    AccessViolation, ChallengeLoader, FOLD_GATE, HashMismatch, NoAdmissibleThreshold, SelectionLoader,
    ThresholdArtifactMissing, freeze_fold_threshold, threshold_grid_hash, verify_challenge_access,
)
from alignair.aa.guard.checkpoint import CheckpointProvenance
from alignair.aa.guard.classes import CLASS_MAP_HASH
from alignair.aa.guard.decision import CandidateSpec, candidate_grid
from alignair.aa.guard.dual_eval import ModeScores, RecordScore, assert_binding_is_blind, binding_scores, dual_eval
from alignair.aa.guard.features import FEATURE_NAMES, hmm_features
from alignair.aa.guard.freeze import feature_order_hash, freeze_pre_unlock_bundle, model_identity, preprocessing_hash
from alignair.aa.guard.fusion import fit_fusion
from alignair.aa.guard.loco import LOCO_FOLDS, LOCO_MASK_HASH, masked_profiles_for
from alignair.aa.guard.model import GuardModel
from alignair.aa.guard.profiles import build_bank_fixture
from alignair.aa.guard.selection import SELECTION_GATES, group_counts
from alignair.aa.guard.thresholds import clopper_pearson_upper
from alignair.aa.guard.tokenizer import AaTokenizer

ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability")
SCRATCH = Path("/tmp/claude-1000/-home-thomas-Desktop-AlignAIR/376842b6-033f-49b5-a34c-02e6621fd496/"
               "scratchpad/mock_oneshot")
LOCO_MAN = json.load(open(ART / "loco_run_manifest.json"))
STAGE4_MAN = json.load(open(ART / "stage4_run_manifest.json"))

IGH_V = b">h1\nQVQLVQSGAEVKKPGASVKVSCKASGYTF\n>h2\nQVQLVQSGAEVKKPGSSVKVSCKASGGTF\n>h3\nEVQLVESGGGLVQPGGSLRLSCAASGFTF\n"
IGH_J = b">hj1\nWGQGTLVTVSSASTKG\n>hj2\nWGQGTTVTVSSASTKG\n>hj3\nWGQGTMVTVSSASTKG\n"
IGK_V = b">k1\nDIQMTQSPSSLSASVGDRVTITCRASQSI\n>k2\nDIQMTQSPSSLSASVGDRVTITCRASQGI\n>k3\nDIQLTQSPSFLSASVGDRVTITCRASQGI\n"
IGK_J = b">kj1\nFGQGTKVEIKRTVA\n>kj2\nFGQGTKLEIKRTVA\n>kj3\nFGGGTKVEIKRTVA\n"
MOCK_FOLD = "igk_igl"
SPEC = CandidateSpec.of("C2")
# selection groups present for the igk_igl fold (the held-out group is excluded from selection)
REQUIRED = ["tcr", "igsf", "structural", "positive_real", "positive_sim"]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _state_hash(model) -> str:
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    return _sha_bytes(buf.getvalue())


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _score(group, *, accept, unit_id, parent=None, within=True):
    """A synthetic RecordScore (prediction + label). accept=True -> the decision graph accepts (supported)."""
    comp = (5.0, 0.0, 0.0, 0.0, 0.0) if accept else (0.0, 5.0, 0.0, 0.0, 0.0)
    typ = (5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if accept else (0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    hmm = {"has_v": True, "has_j": True, "order_ok": True, "coverage_ok": True, "combined_coverage": 0.95,
           "best_v_group": "human_igh" if accept else "human_igl", "best_alt_group": "", "best_j_group": "",
           "human_igh_score": 30.0 if accept else 0.0, "best_alt_score": 0.0 if accept else 30.0,
           "margin": 30.0 if accept else -30.0, "best_v_score": 30.0, "best_j_score": 10.0, "n_significant_hits": 2}
    exp = "supported" if group.startswith("positive") else "unknown_ood"
    return RecordScore(idx=0, record_id_sha256=_sha(unit_id), seq_sha256=_sha(unit_id), raw_length=100,
                       within_length_contract=within,
                       structural_prefilter={"length_contract": "ok" if within else "over_length",
                                             "geometry": "complete"},
                       applicability_logit=(5.0 if accept else -5.0), energy=-10.0,
                       completeness_logits=comp, domain_type_logits=typ, hmm_evidence=hmm, c0_state="",
                       c0_reason="", expected_state=exp, expected_category=group, group=group,
                       unit={"lineage": unit_id, "parent_lineage": parent or unit_id, "donor": "d",
                             "study": "s", "role": "selection"}, masked_groups=())


def _blind(scores):
    return ModeScores(mode="blind", diagnostic_only=False, binding=True, masked_groups=(),
                      input_order_sha256="x", input_contract_digest="c", scores=tuple(scores))


def _passing_selection():
    return _blind([_score("tcr", accept=False, unit_id=f"T{i}") for i in range(80)]
                  + [_score("igsf", accept=False, unit_id=f"S{i}") for i in range(40)]
                  + [_score("structural", accept=False, unit_id=f"st{i}", parent=f"P{i}") for i in range(80)]
                  + [_score("positive_real", accept=True, unit_id=f"R{i}") for i in range(60)]
                  + [_score("positive_sim", accept=True, unit_id=f"M{i}") for i in range(60)])


def _check(checks, name, fn):
    fn()
    checks.append(name)
    print(f"  [ok] {name}", flush=True)


def run_mock() -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    checks: list = []
    demos: dict = {}

    bank = build_bank_fixture(regions={"human_igh": {"v": IGH_V, "j": IGH_J},
                                        "human_igl": {"v": IGK_V, "j": IGK_J}})
    ck = bank.checksum()
    tok = AaTokenizer()
    chal_seqs = [{"aa": "DIQLTQSPSFLSASVGDRVTITCRASQGIAAAAFGGGTKVEIKRTVA", "id": "c1", "group": "igk_igl",
                  "category": "igk", "expected_state": "unknown_ood", "lineage_key": "cl1", "parent_lineage": "cp1"},
                 {"aa": "EVQLVESGGGLVQPGGSLRLSCAASGFTFAAAAWGQGTMVTVSSASTKG", "id": "c2", "group": "positive_real",
                  "category": "pos", "expected_state": "supported", "lineage_key": "cl2", "parent_lineage": "cp2"}]
    vecs = [hmm_features(bank, r["aa"]).to_vector().tolist() for r in chal_seqs]
    fusion = fit_fusion(FEATURE_NAMES, vecs, ck)
    model = GuardModel("C2", len(FEATURE_NAMES), width=16, depth=1, heads=2)

    sel_root, chal_root = SCRATCH / "selection", SCRATCH / "challenge"
    sel_root.mkdir(parents=True, exist_ok=True); chal_root.mkdir(parents=True, exist_ok=True)
    (sel_root / "selection.jsonl").write_text("selection-provenance-only\n")
    chal_bytes_src = ("\n".join(json.dumps(r) for r in chal_seqs) + "\n").encode()
    (chal_root / "challenge.jsonl").write_bytes(chal_bytes_src)
    chal_digest = _sha_bytes(chal_bytes_src)
    sel_man_hash = _sha_bytes((sel_root / "selection.jsonl").read_bytes())

    prov_digest = _sha("mock-provenance")
    file_sha = _state_hash(model)
    model_id = model_identity(prov_digest, file_sha)
    bundle = freeze_pre_unlock_bundle(
        model_hash=model_id, normalization_hash=_sha_bytes(fusion.mean.tobytes() + fusion.std.tobytes()),
        source_vocab_hash=_sha(json.dumps(["cat_a", "cat_b"])), bank_checksum=ck, evaluator_commit=_git_head(),
        cohort_manifest_hash=_sha_bytes((ART / "FROZEN_COHORT_MANIFEST.json").read_bytes()),
        threshold_grid_hash=threshold_grid_hash())

    # ---- P0-3/4 statistical demonstrations (counts DERIVED from synthetic predictions+labels) ----------
    gp0 = candidate_grid("C2")[0]

    def _demo_aggregation():
        recs = [_score("structural", accept=False, unit_id=f"r{i}", parent=f"P{i % 4}") for i in range(100)]
        tbl = group_counts(recs, SPEC, gp0)
        assert SELECTION_GATES["structural"]["unit"] == "parent_lineage"
        assert tbl["structural"].n == 4 and tbl["structural"].k == 0     # 100 records -> 4 parent-lineage units
        demos["parent_lineage_aggregation"] = {"records": 100, "units": tbl["structural"].n}
    _check(checks, "demo_parent_lineage_aggregation", _demo_aggregation)

    def _demo_frr_separation():
        recs = ([_score("positive_real", accept=(i >= 2), unit_id=f"R{i}") for i in range(40)]    # 2 false-rejects
                + [_score("positive_sim", accept=True, unit_id=f"M{i}") for i in range(60)])        # clean
        tbl = group_counts(recs, SPEC, gp0)
        assert not tbl["positive_real"].passed and tbl["positive_sim"].passed   # SEPARATE gates, not pooled
        demos["frr_separation"] = {"real": [tbl["positive_real"].k, tbl["positive_real"].n, tbl["positive_real"].passed],
                                   "sim": [tbl["positive_sim"].k, tbl["positive_sim"].n, tbl["positive_sim"].passed]}
    _check(checks, "demo_real_vs_sim_frr_separation", _demo_frr_separation)

    def _demo_one_group_fail_no_pooling():
        scores = ([_score("tcr", accept=False, unit_id=f"T{i}") for i in range(80)]
                  + [_score("igsf", accept=(i == 0), unit_id=f"S{i}") for i in range(40)])   # 1 false-accept
        tbl = group_counts(scores, SPEC, gp0)
        assert not tbl["igsf"].passed and tbl["tcr"].passed               # igsf gate fails on its own
        pooled_k = sum(g.k for g in tbl.values()); pooled_n = sum(g.n for g in tbl.values())
        pooled_upper = clopper_pearson_upper(pooled_k, pooled_n)
        assert pooled_upper <= 0.08                                       # POOLED would pass...
        from alignair.aa.guard.selection import select_thresholds
        assert select_thresholds(SPEC, scores, ["tcr", "igsf"]) is None   # ...but per-group selection refuses
        demos["one_group_fail_despite_pooled_success"] = {
            "igsf": [tbl["igsf"].k, tbl["igsf"].n, round(tbl["igsf"].upper, 4), tbl["igsf"].passed],
            "pooled_upper": round(pooled_upper, 4), "select_result": "None(STOP)"}
    _check(checks, "demo_one_group_failure_no_pooling", _demo_one_group_fail_no_pooling)

    def _demo_stop():
        always = _blind([_score("tcr", accept=True, unit_id=f"T{i}") for i in range(80)])   # always false-accept
        try:
            freeze_fold_threshold(MOCK_FOLD, "C2", always, ["tcr"], model_identity_hash=model_id,
                                  checkpoint_provenance_digest=prov_digest, checkpoint_file_sha256=file_sha,
                                  selection_manifest_sha256=sel_man_hash, challenge_file_sha256=chal_digest,
                                  frozen_bundle=bundle, out_path=SCRATCH / "stop.json", mock=True)
            raise AssertionError("should have STOPPED")
        except NoAdmissibleThreshold:
            demos["no_admissible_threshold_stop"] = True
    _check(checks, "demo_no_admissible_threshold_stop", _demo_stop)

    # ---- threshold-freeze on the PASSING multi-group selection (derived counts) ------------------------
    thr_path = SCRATCH / "fold_threshold.json"
    sel_scores = _passing_selection()
    _check(checks, "selection_binding_is_blind_only", lambda: (assert_binding_is_blind(sel_scores),
                                                               _raises_ref()))
    artifact = freeze_fold_threshold(MOCK_FOLD, "C2", sel_scores, REQUIRED, model_identity_hash=model_id,
                                     checkpoint_provenance_digest=prov_digest, checkpoint_file_sha256=file_sha,
                                     selection_manifest_sha256=sel_man_hash, challenge_file_sha256=chal_digest,
                                     frozen_bundle=bundle, out_path=thr_path, mock=True, marker="mock_oneshot")

    def _demo_passing():
        assert artifact.is_complete()
        assert set(artifact.per_group) >= set(REQUIRED)
        for g in REQUIRED:
            assert artifact.per_group[g]["passed"] and artifact.per_group[g]["k"] == 0
        demos["passing_multi_group"] = {g: [artifact.per_group[g]["n"], artifact.per_group[g]["passed"]]
                                        for g in REQUIRED}
    _check(checks, "passing_multi_group_threshold_frozen", _demo_passing)

    excluded = (MOCK_FOLD, tuple(sorted(masked_profiles_for(MOCK_FOLD))))

    # ---- BARRIER 1 — selection loader cannot reach a challenge file ------------------------------------
    sel_loader = SelectionLoader(sel_root, chal_root)
    def _b1():
        assert sel_loader.load(sel_root / "selection.jsonl")
        try:
            sel_loader.load(chal_root / "challenge.jsonl"); raise AssertionError("should have refused")
        except AccessViolation:
            pass
    _check(checks, "barrier1_selection_cannot_open_challenge", _b1)

    # ---- BARRIER 2 — challenge refused with no artifact ------------------------------------------------
    def _b2():
        loader = ChallengeLoader(challenge_root=chal_root, selection_root=sel_root,
                                 threshold_artifact_path=SCRATCH / "nope.json", audit_path=SCRATCH / "audit.jsonl",
                                 frozen_bundle=bundle, expected_selection_manifest_sha256=sel_man_hash,
                                 expected_excluded=excluded, expected_challenge_digest=chal_digest)
        try:
            loader.open(chal_root / "challenge.jsonl"); raise AssertionError("opened without artifact")
        except ThresholdArtifactMissing:
            pass
    _check(checks, "barrier2_challenge_refused_without_artifact", _b2)

    # ---- BARRIER 5 — changed weight file hard-fails (identical metadata) -------------------------------
    def _b5():
        tampered = freeze_pre_unlock_bundle(
            model_hash=model_identity(prov_digest, "Z" * 64),
            normalization_hash=bundle.hashes["normalization"], source_vocab_hash=bundle.hashes["source_vocab"],
            bank_checksum=ck, evaluator_commit=_git_head(), cohort_manifest_hash=bundle.hashes["cohort_manifest"],
            threshold_grid_hash=threshold_grid_hash())
        try:
            verify_challenge_access(artifact, tampered, expected_selection_manifest_sha256=sel_man_hash,
                                    expected_excluded=excluded, expected_challenge_digest=chal_digest)
            raise AssertionError("weight drift not caught")
        except HashMismatch:
            pass
    _check(checks, "barrier5_changed_weight_file_hard_fails", _b5)

    # ---- BARRIER 4+7 — atomic audit before scoring; opened bytes verified vs pre-frozen challenge digest
    audit_path = SCRATCH / "audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    chal_loader = ChallengeLoader(challenge_root=chal_root, selection_root=sel_root,
                                  threshold_artifact_path=thr_path, audit_path=audit_path, frozen_bundle=bundle,
                                  expected_selection_manifest_sha256=sel_man_hash, expected_excluded=excluded,
                                  expected_challenge_digest=chal_digest, marker="mock_oneshot")
    chal_data = chal_loader.open(chal_root / "challenge.jsonl")
    def _b47():
        rec = [json.loads(x) for x in audit_path.read_text().splitlines() if x.strip()]
        assert rec[-1]["event"] == "challenge_unlock" and rec[-1]["fold"] == MOCK_FOLD
        assert chal_data == chal_bytes_src
    _check(checks, "barrier4_7_audit_then_verified_challenge_bytes", _b47)

    def _b7_mismatch():
        (chal_root / "tampered.jsonl").write_bytes(b"NOT THE PRE-FROZEN FILE")
        aud = SCRATCH / "audit_fail.jsonl"
        loader = ChallengeLoader(challenge_root=chal_root, selection_root=sel_root, threshold_artifact_path=thr_path,
                                 audit_path=aud, frozen_bundle=bundle, expected_selection_manifest_sha256=sel_man_hash,
                                 expected_excluded=excluded, expected_challenge_digest=chal_digest)
        try:
            loader.open(chal_root / "tampered.jsonl"); raise AssertionError("tampered challenge not caught")
        except HashMismatch:
            pass
        events = [json.loads(x)["event"] for x in aud.read_text().splitlines() if x.strip()]
        assert "challenge_unlock" in events and "challenge_digest_mismatch" in events
    _check(checks, "barrier7_tampered_challenge_writes_failure_audit", _b7_mismatch)

    # ---- dual-EVALUATION on the unlocked challenge (RAW scores; blind masks the fold) ------------------
    chal_recs = [json.loads(x) for x in chal_data.decode().splitlines() if x.strip()]
    chal_res = dual_eval(model, bank, fusion, tok, chal_recs, fold=MOCK_FOLD, candidate="C2", bank_checksum=ck)
    def _b_dual():
        assert chal_res.identical_inputs
        assert chal_res.blind.masked_groups == tuple(sorted(masked_profiles_for(MOCK_FOLD)))
        for sc in chal_res.blind.scores:
            assert "human_igk" not in sc.c0_reason and "human_igl" not in sc.c0_reason
            assert sc.hmm_evidence["best_v_group"] not in ("human_igk", "human_igl")
    _check(checks, "dual_eval_challenge_raw_scores_blind_masked", _b_dual)

    def _b_hashset():
        from alignair.aa.guard.freeze import REQUIRED_SLOTS
        assert set(bundle.hashes) == set(REQUIRED_SLOTS)
        assert artifact.bundle_sha256 == bundle.bundle_sha256()
        assert artifact.grid_digest() == bundle.hashes["threshold_grid"] == threshold_grid_hash()
        assert bundle.hashes["mask_graph"] == LOCO_MASK_HASH and bundle.hashes["class_map"] == CLASS_MAP_HASH
    _check(checks, "full_hash_set_consistent", _b_hashset)

    return {"passed": True, "checks": checks, "n_checks": len(checks), "demonstrations": demos,
            "bundle_sha256": bundle.bundle_sha256(), "artifact_digest": artifact.digest(), "mock_fold": MOCK_FOLD,
            "candidate": "C2", "required_groups": REQUIRED, "challenge_digest": chal_digest,
            "raw_score_sample": {"reference_covered": [s.applicability_logit for s in chal_res.reference_covered.scores],
                                 "blind": [s.applicability_logit for s in chal_res.blind.scores]}}


def _raises_ref():
    """assert reference_covered can never bind (used inline in the binding check)."""
    ref = ModeScores(mode="reference_covered", diagnostic_only=True, binding=False, masked_groups=(),
                     input_order_sha256="x", input_contract_digest="c", scores=())
    try:
        assert_binding_is_blind(ref); raise AssertionError("ref bound")
    except ValueError:
        return True


def _model_digests() -> dict:
    """Each LOCO checkpoint's provenance digest, ACTUAL file SHA-256, and normalization hash (burned-dev)."""
    out = {}
    for p in sorted(glob.glob(str(LOCO_MAN["checkpoint_dir"]) + "/*.final.pt")):
        raw = Path(p).read_bytes()
        ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        prov, be = ck["provenance"], ck["bound_extra"]
        cp = CheckpointProvenance(tokenizer=prov["tokenizer"], bank_checksum=prov["bank_checksum"],
                                  feature_order=tuple(prov["feature_order"]), class_map_hash=prov["class_map_hash"],
                                  loss_mask_config=prov["loss_mask_config"],
                                  training_manifest_hash=prov["training_manifest_hash"],
                                  partition_hash=prov["partition_hash"], seed=prov["seed"],
                                  software_versions=prov["software_versions"])
        file_sha = _sha_bytes(raw)
        cid, _seed, fold = Path(p).stem.replace(".final", "").split(".")
        out[f"{cid}.{fold}"] = {"candidate": cid, "fold": fold,
                                "checkpoint_provenance_digest": cp.digest(), "checkpoint_file_sha256": file_sha,
                                "model_identity": model_identity(cp.digest(), file_sha),
                                "normalization_stats_sha256": be["normalization_stats_sha256"],
                                "partition_hash": be["partition_hash"]}
    return out


def write_pre_unlock_manifest() -> dict:
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
                    "excluded_group": g, "masked_profiles": sorted(masked_profiles_for(g)),
                    "challenge_file_sha256": "PENDING_hydration"} for g in LOCO_FOLDS}
    bundles = {}
    for key, m in per_model.items():
        b = freeze_pre_unlock_bundle(
            model_hash=m["model_identity"], normalization_hash=m["normalization_stats_sha256"],
            source_vocab_hash=per_fold[m["fold"]]["source_vocab"], bank_checksum=glob_hashes["profile_bank"],
            evaluator_commit=glob_hashes["evaluator_commit"], cohort_manifest_hash=glob_hashes["cohort_manifest"],
            threshold_grid_hash=glob_hashes["threshold_grid"])
        bundles[key] = b.bundle_sha256()
    manifest = {
        "artifact": "pre_unlock_frozen_manifest", "immutable": True,
        "purpose": "Every hash frozen BEFORE the first fresh-selection read. The fold-threshold artifact "
                   "produced on fresh selection, and every challenge unlock, is verified against these.",
        "candidate_decision_components": {c: list(CandidateSpec.of(c).components) for c in ("C0", "C1", "C2", "C3", "C4")},
        "candidate_grid_sizes": {c: len(candidate_grid(c)) for c in ("C0", "C1", "C2", "C3", "C4")},
        "selection_gates": SELECTION_GATES, "fold_gate": FOLD_GATE,
        "challenge_binding_contract": "each fold's role-specific challenge file digest is registered at fresh "
                                      "hydration (blind, contents not inspected) and re-verified against the "
                                      "opened bytes after the unlock audit; a mismatch writes a failure audit "
                                      "and refuses. Value is PENDING until hydration (no fresh data opened now).",
        "model_identity_rule": "sha256({provenance_digest, checkpoint_file_sha256}) — a weight-file change under "
                               "identical metadata drifts the model slot and hard-fails.",
        "global": glob_hashes, "per_fold": per_fold, "per_model": per_model,
        "per_model_bundle_sha256": bundles,
        "not_opened": ["fresh_selection", "loco_challenge", "far_ood", "sealed locked_test.jsonl"],
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (ART / "PRE_UNLOCK_FROZEN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main():
    print("[mock] running synthetic one-shot (derived counts, corrected contract) ...", flush=True)
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
