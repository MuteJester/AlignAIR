"""Task 9 (final) — AA inference PRODUCT-PATH validation harness (spec §11 Slice 5).

Runs the real product path (`load_aa_model` -> `AAPredictor.predict`) — never a shortcut through
`alignair.aa.strategic`/GenAIRR-derived class maps directly — and checks it against the independently
reconstructed research path, a frozen simulator golden cohort, and the frozen Phase-3 16-antibody
cohort. Writes `docs/superpowers/artifacts/2026-07-23-aa-product/validation.json` (pass/fail per check)
+ `validation.md` (short human summary). This is a SCRIPT that produces artifacts, not a pytest suite —
it reuses existing product/research code (`alignair.aa.product.*`, `alignair.aa.infer.class_probs`,
`alignair.aa.conformal.FrozenConformal`) rather than reimplementing any conformal/expansion logic.

Six checks (spec §11 Slice 5 / task-9 brief):
  1. golden_cohort       — frozen simulator cohort spanning full/trunc/short coverage classes
                           (fixed seed disjoint from prior AA cohort seeds), with known true V/D/J.
  2. parity              — product-path V/D/J top_class + calibrated_class_set == research path
                           (`class_probs` + `FrozenConformal` reconstructed from the bundle's own raw
                           calibration bytes), exactly, on every read x gene in the golden cohort.
  3. coordinates         — §5: coordinates within [0, len] or null; V_end<=J_start -> junction_aa ==
                           sequence[V_end:J_start]; start>end voiding verified end-to-end.
  4. groups              — §7: coverage_group=None -> pooled_fallback; coverage_group="full" ->
                           supplied (+ warning), and the supplied set matches q_mondrian["full"].
  5. phase3_descriptive  — frozen Phase-3 16-antibody cohort run through the PRODUCT path; reports
                           top-class IMGT V-family accuracy. DESCRIPTIVE ONLY — no gate/promotion.
                           Skips cleanly (does not fail) if the git-ignored hydration file is absent.
  6. tamper_rejection    — a byte-flipped section, and a bundle whose calibration
                           provenance.model_sha256 != the card's source_model_sha256, both make
                           `load_aa_model` raise.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/product/validate_product.py --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import GenAIRR as ga
from alignair_benchmark import default_igh_spec, stream_benchmark

from alignair.aa.augment import aa_truncate
from alignair.aa.conformal import FrozenConformal
from alignair.aa.equivalence import EquivalenceClassMap
from alignair.aa.gym_bridge import aa_usable
from alignair.aa.infer import class_probs
from alignair.aa.phase3.cohort import load_hydrated, load_labels
from alignair.aa.product.bundle import complete_calibration, save_aa_bundle
from alignair.aa.product.calibration_group import SUPPLIED_WARNING
from alignair.aa.product.loader import AAPredictor, _resolve_path, load_aa_model
from alignair.aa.product.schema import AAPrediction
from alignair.aa.product.validate import VALID_AMINO_ACIDS
from alignair.model_file import container

REPO_ROOT = Path(__file__).resolve().parents[3]
ART_DIR = REPO_ROOT / "docs" / "superpowers" / "artifacts" / "2026-07-23-aa-product"
PHASE3_DIR = REPO_ROOT / "docs" / "superpowers" / "artifacts" / "2026-07-21-aa-phase3"

DEFAULT_MODEL = "alignair-aa-igh-human@2.0.0"
DC_NAME = "HUMAN_IGH_OGRDB"
# Fixed, disjoint from every prior AA cohort seed used elsewhere in this project (eval_strategic.py
# uses 31; evaluator.py's build_cohort uses 20250722; calibrate_stage2.py uses 1000/9000).
COHORT_SEED = 20260201

# The SAME coverage-bin definitions `scripts/aa/calibrate_stage2.py` used to fit the frozen
# full/trunc/short Mondrian quantiles embedded in the v2 bundle -- so the golden cohort's "trunc"/
# "short" reads are the honest population those quantiles were calibrated against, not an arbitrary
# re-guess of the bin boundaries.
_TRUNC_KEEP_RANGE = (0.6, 0.9)
_SHORT_KEEP_RANGE = (0.3, 0.6)


def _first(call) -> str:
    return str(call or "").split(",")[0].strip()


# ============================================================== Check 1: frozen golden cohort


def build_golden_cohort(predictor: AAPredictor, *, dc_name: str = DC_NAME, n_per_stratum: int = 60,
                        seed: int = COHORT_SEED, n_target_full: int = 15) -> list[dict]:
    """One seeded `stream_benchmark` pass over `dc_name`'s `clean_full` stratum (forward orientation,
    AA-usable, known V+J allele) -> up to `n_target_full` clean full-length reads, each expanded into
    THREE golden-cohort rows: `full` (uncropped), `trunc` (`aa_truncate` keep in [0.6,0.9]), `short`
    (keep in [0.3,0.6)) -- see `_TRUNC_KEEP_RANGE`/`_SHORT_KEEP_RANGE`. Deterministic given
    `(dc_name, n_per_stratum, seed, n_target_full)`: `stream_benchmark` itself is seeded, and the
    truncation RNG is a fresh `np.random.default_rng(seed)` consumed in the same fixed order every
    call (verified directly by `check_golden_cohort`'s rebuild-and-compare)."""
    dc = getattr(ga, dc_name)
    vclass = predictor.class_maps["V"].allele_class
    jclass = predictor.class_maps["J"].allele_class
    rng = np.random.default_rng(seed)
    spec = default_igh_spec(n_per_stratum=n_per_stratum, seed=seed)

    pool = []
    for c in stream_benchmark(spec, dataconfig=dc):
        if c.stratum != "clean_full" or c.orientation_id != 0:
            continue
        r = c.record
        if not aa_usable(r):
            continue
        vc, jc = _first(r["v_call"]), _first(r["j_call"])
        if vc not in vclass or jc not in jclass:
            continue
        aa = r["sequence_aa"]
        # `aa_usable` only guarantees a translation EXISTS, not that it is a clean, stop-free product-
        # path-valid sequence (GenAIRR occasionally emits a translated read whose frame runs into a
        # stop codon outside the V..J span) -- skip anything the product's own §9.1 alphabet would
        # reject, so the golden cohort never trips over `predict(on_invalid="error")`'s stop-codon gate.
        if not aa or any(ch not in VALID_AMINO_ACIDS for ch in aa):
            continue
        pool.append({"aa": aa, "v_call": vc, "j_call": jc, "d_call": _first(r.get("d_call"))})
        if len(pool) >= n_target_full:
            break

    cohort = []
    for i, row in enumerate(pool):
        base = {"true_v_call": row["v_call"], "true_j_call": row["j_call"],
               "true_d_call": row["d_call"], "lineage": i}
        cohort.append({"sequence_id": f"golden_{i}_full", "sequence": row["aa"],
                       "coverage_class": "full", **base})
        keep_trunc = float(rng.uniform(*_TRUNC_KEEP_RANGE))
        cohort.append({"sequence_id": f"golden_{i}_trunc",
                       "sequence": aa_truncate(row["aa"], keep_trunc, keep_trunc, rng),
                       "coverage_class": "trunc", **base})
        keep_short = float(rng.uniform(*_SHORT_KEEP_RANGE))
        cohort.append({"sequence_id": f"golden_{i}_short",
                       "sequence": aa_truncate(row["aa"], keep_short, keep_short, rng),
                       "coverage_class": "short", **base})
    return cohort


def check_golden_cohort(predictor: AAPredictor, cohort: list[dict], *, dc_name: str,
                        n_per_stratum: int, seed: int, n_target_full: int) -> dict:
    n_full = sum(1 for r in cohort if r["coverage_class"] == "full")
    n_trunc = sum(1 for r in cohort if r["coverage_class"] == "trunc")
    n_short = sum(1 for r in cohort if r["coverage_class"] == "short")
    vclass = predictor.class_maps["V"].allele_class
    jclass = predictor.class_maps["J"].allele_class
    truth_valid = all(r["true_v_call"] in vclass and r["true_j_call"] in jclass for r in cohort)

    # Determinism ("frozen"): rebuild from scratch with the identical knobs and require a byte-for-byte
    # identical cohort (ids, sequences, and truth calls) -- not merely "we passed a seed".
    rebuilt = build_golden_cohort(predictor, dc_name=dc_name, n_per_stratum=n_per_stratum, seed=seed,
                                  n_target_full=n_target_full)
    deterministic = (
        [r["sequence_id"] for r in cohort] == [r["sequence_id"] for r in rebuilt]
        and [r["sequence"] for r in cohort] == [r["sequence"] for r in rebuilt]
        and [r["true_v_call"] for r in cohort] == [r["true_v_call"] for r in rebuilt]
        and [r["true_j_call"] for r in cohort] == [r["true_j_call"] for r in rebuilt]
    )

    pass_ = bool(n_full > 0 and n_trunc > 0 and n_short > 0 and truth_valid and deterministic)
    return {"pass": pass_, "detail": {
        "seed": seed, "dc_name": dc_name, "n_per_stratum": n_per_stratum,
        "n_target_full_reads": n_target_full,
        "n_full": n_full, "n_trunc": n_trunc, "n_short": n_short, "n_total": len(cohort),
        "truth_labels_present_in_bundle_class_maps": truth_valid,
        "deterministic_rebuild": deterministic,
        "coverage_bin_defs": (
            "full=uncropped; trunc=aa_truncate keep in [0.6,0.9]; short=aa_truncate keep in [0.3,0.6) "
            "-- matches scripts/aa/calibrate_stage2.py's Mondrian bin definitions"),
    }}


# ============================================================== Check 2: parity


def check_parity(predictor: AAPredictor, cohort: list[dict], raw_calibration: dict) -> dict:
    """Product-path V/D/J top_class + calibrated_class_set (pooled default) vs the independently
    reconstructed research path: `class_probs(predictor.model, ...)` (a fresh forward pass, NOT
    `predictor.predict`'s internal one) + `FrozenConformal` built straight from the bundle's own raw
    `calibration` section bytes (read via `container.read_section`, bypassing `predictor.conformal`
    entirely) -- so this is a genuinely independent recomputation, not a self-check against the same
    object the product path already uses."""
    cal = complete_calibration(raw_calibration)
    method = cal["method"]
    lam, k_reg = method["lambda"], method["k_reg"]
    genes = ("V", "J", "D") if getattr(predictor.cfg, "has_d", False) else ("V", "J")
    gene_key = {"V": "v_class_logits", "J": "j_class_logits", "D": "d_class_logits"}

    seqs = [r["sequence"] for r in cohort]
    batch = [{"sequence_id": r["sequence_id"], "sequence": r["sequence"]} for r in cohort]
    product_results = predictor.predict(batch, coverage_group=None)
    bad = [res.sequence_id for res in product_results if not isinstance(res, AAPrediction)]
    if bad:
        raise RuntimeError(f"check_parity: golden cohort produced ErrorResult(s), expected all-valid "
                          f"AA sequences: {bad}")

    mismatches = []
    n_checks = 0
    for gene in genes:
        probs = class_probs(predictor.model, seqs, predictor.cfg, predictor.device, gene_key[gene])
        q_pooled = cal["genes"][gene]["calibration_frozen"]["q_pooled"]
        fc = FrozenConformal(q_pooled, lam=lam, k_reg=k_reg)
        for row, result, prob_row in zip(cohort, product_results, probs):
            n_checks += 1
            research_top = int(prob_row.argmax())
            research_set = sorted(int(x) for x in fc.predict_set(prob_row))
            product_top = result.segments[gene]["top_call"]["class_id"]
            product_set = sorted(int(x) for x in result.segments[gene]["calibrated_class_set"])
            if research_top != product_top or research_set != product_set:
                mismatches.append({"sequence_id": row["sequence_id"], "gene": gene,
                                   "research_top": research_top, "product_top": product_top,
                                   "research_set": research_set, "product_set": product_set})

    return {"pass": len(mismatches) == 0, "detail": {
        "n_reads": len(cohort), "genes": list(genes), "n_checks": n_checks,
        "n_mismatches": len(mismatches), "mismatches": mismatches[:20],
    }}


# ============================================================== Check 3: coordinates + junction


def _minimal_class_map(gene: str, allele: str) -> EquivalenceClassMap:
    return EquivalenceClassMap(gene=gene, db_hash="fakehash", allele_class={allele: 0},
                               class_members={0: [allele]})


class _StubModel:
    """Ignores its input, always returns a fixed forward-pass output -- used to force a deterministic
    coordinate contract without needing real model weights (mirrors
    tests/alignair/aa/product/test_predictor.py's `_StubModel`)."""

    def __init__(self, out: dict):
        self._out = out

    def eval(self):
        return self

    def __call__(self, batch):
        return self._out


class _StubTokenizer:
    def encode(self, sequence: str):
        return np.ones(len(sequence), dtype=np.int64)


class _StubCfg:
    max_seq_length = 32
    has_d = True


def _voiding_contract_predictor(out: dict) -> AAPredictor:
    return AAPredictor(
        model=_StubModel(out), cfg=_StubCfg(), tokenizer=_StubTokenizer(),
        class_maps={"V": _minimal_class_map("V", "IGHV1-1*01"), "J": _minimal_class_map("J", "IGHJ1*01")},
        d_allele_order=["IGHD1-1*01"],
        asc_multi={"IGHV1-1*01": ["IGHV1-1*01"], "IGHJ1*01": ["IGHJ1*01"]},
        conformal={g: {"pooled": FrozenConformal(0.99), "mondrian": {}} for g in ("V", "J", "D")},
        card={"model_id": "validate-product-harness-stub", "model_version": "0.0.0-harness",
             "alignair_version": "harness-test", "calibration_scope": "HARNESS STUB SCOPE STRING",
             "integrity": {"weights_sha256": "abc", "calibration_sha256": "def"},
             "reference_provenance": {"class_map_db_hash": {"V": "v", "J": "j"}}},
        device="cpu",
    )


def check_start_gt_end_voiding_contract() -> tuple[bool, dict]:
    """Directly exercises §5's start>end voiding + clipping rules end-to-end through the real
    `AAPredictor.predict` path (mirrors `test_predictor.py`'s coordinate tests), on a crafted
    deterministic stub -- the real v2 model essentially never produces an incoherent start>end on a
    clean golden-cohort read, so this is how the invariant gets exercised without depending on the
    model happening to fail in exactly the right way."""
    out = {
        "v_class_logits": torch.tensor([[2.0]]), "j_class_logits": torch.tensor([[2.0]]),
        "d_class_logits": torch.tensor([[2.0]]),
        "v_start": torch.tensor([[8.0]]), "v_end": torch.tensor([[3.0]]),    # start>end -> void
        "j_start": torch.tensor([[5.0]]), "j_end": torch.tensor([[9.0]]),
        "d_start": torch.tensor([[4.0]]), "d_end": torch.tensor([[6.0]]),
    }
    [result] = _voiding_contract_predictor(out).predict(["ACDEFGHIKL"])       # length 10
    voided_ok = (result.segments["V"]["coordinates"] is None
                and "V: incoherent start>end" in result.warnings
                and result.segments["J"]["coordinates"] == {"start": 5, "end": 9}
                and result.junction_aa is None)

    out2 = dict(out)
    out2["v_start"], out2["v_end"] = torch.tensor([[-5.0]]), torch.tensor([[999.0]])   # clip both ways
    [result2] = _voiding_contract_predictor(out2).predict(["ACDEFGHIKL"])
    clipping_ok = result2.segments["V"]["coordinates"] == {"start": 0, "end": 10}

    return bool(voided_ok and clipping_ok), {"voided_ok": bool(voided_ok), "clipping_ok": bool(clipping_ok)}


def check_coordinates(cohort: list[dict], product_results: list) -> dict:
    bounds_ok = True
    junction_ok = True
    n_segments_checked = 0
    n_voided_organic = 0
    violations = []

    for row, result in zip(cohort, product_results):
        d = result.to_dict()
        seq_len = len(row["sequence"])
        for gene, seg in d["segments"].items():
            n_segments_checked += 1
            coords = seg["coordinates"]
            if coords is None:
                if any(f"{gene}: incoherent start>end" in w for w in d["warnings"]):
                    n_voided_organic += 1
                continue
            s, e = coords["start"], coords["end"]
            if not (isinstance(s, int) and isinstance(e, int) and 0 <= s <= e <= seq_len):
                bounds_ok = False
                violations.append({"sequence_id": d["sequence_id"], "gene": gene, "coords": coords,
                                   "seq_len": seq_len})

        v = d["segments"].get("V", {}).get("coordinates")
        j = d["segments"].get("J", {}).get("coordinates")
        derivable = bool(v and j and v["end"] <= j["start"])
        if derivable:
            expected = row["sequence"][v["end"]:j["start"]]
            if d["junction_aa"] != expected:
                junction_ok = False
                violations.append({"sequence_id": d["sequence_id"], "issue": "junction_aa mismatch",
                                   "expected_len": len(expected), "got_len": len(d["junction_aa"] or "")})
        elif d["junction_aa"] is not None:
            junction_ok = False
            violations.append({"sequence_id": d["sequence_id"],
                               "issue": "junction_aa should be null when V/J not both coherent"})

    voiding_pass, voiding_detail = check_start_gt_end_voiding_contract()
    pass_ = bounds_ok and junction_ok and voiding_pass
    return {"pass": pass_, "detail": {
        "n_reads": len(cohort), "n_segments_checked": n_segments_checked,
        "bounds_ok": bounds_ok, "junction_ok": junction_ok,
        "n_voided_organic_in_cohort": n_voided_organic,
        "voiding_contract": voiding_detail,
        "violations": violations[:20],
    }}


# ============================================================== Check 4: pooled vs supplied group


def check_groups(predictor: AAPredictor, cohort: list[dict], raw_calibration: dict) -> dict:
    batch = [{"sequence_id": r["sequence_id"], "sequence": r["sequence"]} for r in cohort]
    pooled = predictor.predict(batch, coverage_group=None)
    supplied = predictor.predict(batch, coverage_group="full")

    pooled_ok = all(p.applied_calibration_group == {"group": "pooled", "source": "pooled_fallback"}
                    for p in pooled)
    supplied_ok = all(p.applied_calibration_group == {"group": "full", "source": "supplied"}
                      for p in supplied)
    warning_ok = all(SUPPLIED_WARNING in p.warnings for p in supplied)

    cal = complete_calibration(raw_calibration)
    method = cal["method"]
    lam, k_reg = method["lambda"], method["k_reg"]
    seqs = [r["sequence"] for r in cohort]
    probs_v = class_probs(predictor.model, seqs, predictor.cfg, predictor.device, "v_class_logits")
    q_pooled = cal["genes"]["V"]["calibration_frozen"]["q_pooled"]
    q_full = cal["genes"]["V"]["calibration_frozen"]["q_mondrian"]["full"]
    fc_full = FrozenConformal(q_full, lam=lam, k_reg=k_reg)
    fc_pooled = FrozenConformal(q_pooled, lam=lam, k_reg=k_reg)

    set_match_ok = True
    n_diff_from_pooled = 0
    for p, prob_row in zip(supplied, probs_v):
        research_full = sorted(int(x) for x in fc_full.predict_set(prob_row))
        product_full = sorted(int(x) for x in p.segments["V"]["calibrated_class_set"])
        if research_full != product_full:
            set_match_ok = False
        research_pooled = sorted(int(x) for x in fc_pooled.predict_set(prob_row))
        if research_full != research_pooled:
            n_diff_from_pooled += 1

    pass_ = pooled_ok and supplied_ok and warning_ok and set_match_ok
    return {"pass": pass_, "detail": {
        "n_reads": len(cohort),
        "pooled_labeled_pooled_fallback": pooled_ok,
        "supplied_labeled_supplied": supplied_ok,
        "supplied_warning_present": warning_ok,
        "supplied_full_v_set_matches_q_mondrian_full": set_match_ok,
        "q_pooled_V": q_pooled, "q_mondrian_full_V": q_full,
        "n_reads_where_full_set_differs_from_pooled_set": n_diff_from_pooled,
    }}


# ============================================================== Check 5: 16-Ab Phase-3 descriptive


def check_phase3_descriptive(predictor: AAPredictor) -> dict:
    gold_path = PHASE3_DIR / "gold_labels.tsv"
    hydrated_path = PHASE3_DIR / "hydrated_sequences.tsv"
    if not hydrated_path.exists():
        return {"pass": True, "detail": {"status": "skipped: hydration not present"}}

    gold, _audit = load_labels(str(gold_path))
    hydrated = load_hydrated(str(hydrated_path))

    rows = [(r, hydrated.get(r.exact_vh_id)) for r in gold]
    missing = [r.name for r, vh in rows if vh is None]
    scoreable = [(r, vh) for r, vh in rows if vh is not None]

    batch = [{"sequence_id": r.name, "sequence": vh} for r, vh in scoreable]
    results = predictor.predict(batch, coverage_group=None) if batch else []
    results_by_id = {res.sequence_id: res for res in results}

    hits, misses = 0, []
    for r, _vh in scoreable:
        res = results_by_id.get(r.name)
        if not isinstance(res, AAPrediction):
            misses.append({"name": r.name, "issue": "predict returned ErrorResult",
                          "detail": getattr(res, "detail", None)})
            continue
        families = res.segments["V"]["top_call"]["families"]
        if r.true_v_family in families:
            hits += 1
        else:
            misses.append({"name": r.name, "true_v_family": r.true_v_family,
                           "predicted_families": families})

    n = len(scoreable)
    accuracy = (hits / n) if n else None
    return {"pass": True, "detail": {
        "status": "ran",
        "n_gold": len(gold), "n_scored": n, "n_missing_hydration": len(missing),
        "hits": hits, "family_accuracy_fraction": f"{hits}/{n}" if n else "0/0",
        "family_accuracy": accuracy, "misses": misses,
        "note": ("DESCRIPTIVE ONLY -- no gate/promotion decision. This re-confirms the PRODUCT path "
                "(load_aa_model -> AAPredictor.predict, one-to-many ASC->IMGT bridge) reproduces the "
                "family-accuracy anchor already established by the research path in phase3_eval.json; "
                "Phase-3's overall decision remains 'inconclusive' (see PHASE3.md) regardless of this "
                "number."),
    }}


# ============================================================== Check 6: tamper + wrong-model


def _flip_byte_in_section(path: Path, out_path: Path, section_name: str) -> None:
    """Flip one byte inside a section's COMPRESSED bytes on disk -- `container.read_section`'s own
    checksum re-check should raise before any loader-specific logic runs (mirrors
    `tests/alignair/aa/product/test_loader.py::_flip_byte_in_section`)."""
    header = container.read_header(str(path))
    s = header["sections"][section_name]
    base = header["_sections_base"]
    data = bytearray(open(path, "rb").read())
    offset = base + s["offset"]
    data[offset] ^= 0xFF
    with open(out_path, "wb") as f:
        f.write(bytes(data))


def check_tamper_and_wrong_model(predictor: AAPredictor, raw_calibration: dict, tmp_dir: str) -> dict:
    tmp_dir_path = Path(tmp_dir)
    model_sha = raw_calibration["provenance"]["model_sha256"]
    card_extra = {"model_id": "validate-product-harness-test", "model_version": "0.0.0-harness",
                 "calibration_scope": predictor.card.get("calibration_scope") or "HARNESS TEST SCOPE",
                 "license": "GPL-3.0-or-later", "source_model_sha256": model_sha}

    # ---- baseline: re-save the ALREADY-LOADED, verified model/calibration as a fresh tmp bundle ----
    # (reuses the loaded predictor's own components instead of re-deriving class maps from GenAIRR via
    # strategic_setup, which is slow and unnecessary here -- the bundle round-trips the same content).
    good_path = tmp_dir_path / "harness_good.alignair"
    save_aa_bundle(str(good_path), state_dict=predictor.model.state_dict(), cfg=predictor.cfg,
                  class_maps=predictor.class_maps, d_allele_order=predictor.d_allele_order,
                  asc_multi=predictor.asc_multi, calibration=raw_calibration, card_extra=card_extra)
    good_load_ok, good_load_error = False, None
    try:
        load_aa_model(str(good_path))
        good_load_ok = True
    except Exception as e:      # noqa: BLE001 -- we want to record ANY unexpected failure here
        good_load_error = str(e)

    # ---- (a) tamper: byte-flip the weights section -> integrity check must raise ----
    tampered_path = tmp_dir_path / "harness_tampered.alignair"
    _flip_byte_in_section(good_path, tampered_path, "weights")
    tamper_raised, tamper_error = False, None
    try:
        load_aa_model(str(tampered_path))
    except ValueError as e:
        tamper_raised, tamper_error = True, str(e)

    # ---- (b) wrong-model: calibration provenance.model_sha256 != card source_model_sha256 ----
    wrong_calibration = copy.deepcopy(raw_calibration)
    fake_sha = "f" * 64 if model_sha != "f" * 64 else "e" * 64
    wrong_calibration["provenance"]["model_sha256"] = fake_sha
    wrong_path = tmp_dir_path / "harness_wrong_model.alignair"
    save_aa_bundle(str(wrong_path), state_dict=predictor.model.state_dict(), cfg=predictor.cfg,
                  class_maps=predictor.class_maps, d_allele_order=predictor.d_allele_order,
                  asc_multi=predictor.asc_multi, calibration=wrong_calibration,
                  card_extra=dict(card_extra, model_id="validate-product-harness-wrong-model-test"))
    wrong_model_raised, wrong_model_error = False, None
    try:
        load_aa_model(str(wrong_path))
    except ValueError as e:
        wrong_model_raised, wrong_model_error = True, str(e)

    pass_ = good_load_ok and tamper_raised and wrong_model_raised
    return {"pass": pass_, "detail": {
        "good_bundle_loads": good_load_ok, "good_bundle_load_error": good_load_error,
        "tamper_flip_byte_in_weights_raises": tamper_raised, "tamper_error": tamper_error,
        "wrong_model_calibration_binding_raises": wrong_model_raised,
        "wrong_model_error": wrong_model_error,
    }}


# ============================================================== reporting + main


_ORDER = ("golden_cohort", "parity", "coordinates", "groups", "phase3_descriptive", "tamper_rejection")


def render_markdown(result: dict) -> str:
    lines = [
        "# AA inference product-path validation (Task 9)", "",
        f"Model: `{result['model']}` &middot; device: `{result['device']}` &middot; "
        f"cohort seed: `{result['cohort_seed']}` &middot; n_reads: {result['n_reads']}", "",
        f"**ALL PASS: {result['all_pass']}**", "",
        "| Check | Result | Summary |", "|---|---|---|",
    ]

    c = result["checks"]
    g, p, co, gr, ph, ta = (c[k]["detail"] for k in _ORDER)
    summaries = {
        "golden_cohort": f"{g['n_full']} full / {g['n_trunc']} trunc / {g['n_short']} short reads; "
                         f"deterministic rebuild: {g['deterministic_rebuild']}",
        "parity": f"{p['n_checks']} (read x gene) checks over {p['n_reads']} reads x "
                  f"{len(p['genes'])} genes ({', '.join(p['genes'])}); {p['n_mismatches']} mismatches",
        "coordinates": f"bounds_ok={co['bounds_ok']} junction_ok={co['junction_ok']} "
                       f"voiding_contract={co['voiding_contract']}",
        "groups": f"pooled/supplied labels + warning correct; supplied 'full' V-set == q_mondrian['full']: "
                  f"{gr['supplied_full_v_set_matches_q_mondrian_full']} "
                  f"({gr['n_reads_where_full_set_differs_from_pooled_set']}/{gr['n_reads']} reads differ "
                  f"from the pooled set)",
        "phase3_descriptive": (ph["status"] if ph["status"].startswith("skipped")
                               else f"{ph['family_accuracy_fraction']} ({ph['family_accuracy']:.3f})"),
        "tamper_rejection": f"tamper raises={ta['tamper_flip_byte_in_weights_raises']}; "
                            f"wrong-model raises={ta['wrong_model_calibration_binding_raises']}",
    }
    for name in _ORDER:
        status = "PASS" if c[name]["pass"] else "FAIL"
        lines.append(f"| `{name}` | {status} | {summaries[name]} |")
    lines.append("")

    if ph["status"] == "ran":
        lines += ["## Phase-3 16-antibody descriptive family accuracy", "",
                  f"{ph['family_accuracy_fraction']} ({ph['family_accuracy']:.3f}) over "
                  f"{ph['n_scored']}/{ph['n_gold']} scoreable gold rows "
                  f"({ph['n_missing_hydration']} missing hydration). DESCRIPTIVE ONLY — no gate/"
                  "promotion decision; Phase-3's frozen decision remains `inconclusive` regardless "
                  "(see `../2026-07-21-aa-phase3/PHASE3.md`).", ""]
        if ph["misses"]:
            lines.append("Misses:")
            for m in ph["misses"]:
                lines.append(f"- {m}")
            lines.append("")

    lines.append("Full detail (including per-check provenance) in `validation.json`.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=COHORT_SEED)
    ap.add_argument("--dc", default=DC_NAME)
    ap.add_argument("--n-per-stratum", type=int, default=60)
    ap.add_argument("--n-target-full", type=int, default=15)
    ap.add_argument("--out-dir", default=str(ART_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] loading product model {args.model!r} (device={args.device}) ...")
    predictor = load_aa_model(args.model, device=args.device)
    bundle_path = _resolve_path(args.model)
    raw_calibration = json.loads(container.read_section(bundle_path, "calibration"))
    print(f"    OK: V/J/D classes = {predictor.cfg.v_allele_count}/{predictor.cfg.j_allele_count}/"
         f"{predictor.cfg.d_allele_count}")

    print("[2/7] building frozen golden simulator cohort (full/trunc/short) ...")
    cohort = build_golden_cohort(predictor, dc_name=args.dc, n_per_stratum=args.n_per_stratum,
                                 seed=args.seed, n_target_full=args.n_target_full)
    if not cohort:
        raise RuntimeError("golden cohort build produced zero usable reads -- increase --n-per-stratum")
    print(f"    OK: {len(cohort)} reads")

    checks: dict = {}

    checks["golden_cohort"] = check_golden_cohort(predictor, cohort, dc_name=args.dc,
                                                  n_per_stratum=args.n_per_stratum, seed=args.seed,
                                                  n_target_full=args.n_target_full)
    print(f"    golden_cohort: {'PASS' if checks['golden_cohort']['pass'] else 'FAIL'}")

    print("[3/7] V/D/J call + calibrated-set parity (product vs research path) ...")
    checks["parity"] = check_parity(predictor, cohort, raw_calibration)
    pd_ = checks["parity"]["detail"]
    print(f"    parity: {'PASS' if checks['parity']['pass'] else 'FAIL'} "
         f"({pd_['n_checks']} checks, {pd_['n_mismatches']} mismatches)")

    print("[4/7] coordinate + junction checks ...")
    batch = [{"sequence_id": r["sequence_id"], "sequence": r["sequence"]} for r in cohort]
    pooled_results = predictor.predict(batch, coverage_group=None)
    checks["coordinates"] = check_coordinates(cohort, pooled_results)
    print(f"    coordinates: {'PASS' if checks['coordinates']['pass'] else 'FAIL'}")

    print("[5/7] pooled vs supplied-group behavior ...")
    checks["groups"] = check_groups(predictor, cohort, raw_calibration)
    print(f"    groups: {'PASS' if checks['groups']['pass'] else 'FAIL'}")

    print("[6/7] 16-antibody Phase-3 descriptive product-path family accuracy ...")
    checks["phase3_descriptive"] = check_phase3_descriptive(predictor)
    p3 = checks["phase3_descriptive"]["detail"]
    print(f"    phase3_descriptive: {p3.get('status')} {p3.get('family_accuracy_fraction', '')}".rstrip())

    print("[7/7] tamper + wrong-model rejection ...")
    with tempfile.TemporaryDirectory(prefix="aa_product_validate_") as tmp_dir:
        checks["tamper_rejection"] = check_tamper_and_wrong_model(predictor, raw_calibration, tmp_dir)
    print(f"    tamper_rejection: {'PASS' if checks['tamper_rejection']['pass'] else 'FAIL'}")

    all_pass = all(c["pass"] for c in checks.values())
    result = {
        "model": args.model, "device": args.device, "cohort_seed": args.seed,
        "n_reads": len(cohort), "dc_name": args.dc, "checks": checks, "all_pass": all_pass,
    }

    out_json = out_dir / "validation.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    out_md = out_dir / "validation.md"
    with open(out_md, "w") as f:
        f.write(render_markdown(result))

    print(f"\n{'=' * 70}\nALL PASS: {all_pass}\nwrote {out_json}\nwrote {out_md}\n{'=' * 70}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
