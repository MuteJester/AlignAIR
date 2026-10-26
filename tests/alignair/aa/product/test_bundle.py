import pytest
import json

import GenAIRR as ga

from alignair.aa.product.asc import asc_to_imgt_multi
from alignair.aa.product.bundle import canonical_json, complete_calibration, save_aa_bundle
from alignair.aa.strategic import strategic_setup
from alignair.model_file import container

pytestmark = pytest.mark.component

_CALIBRATION = {
    "alpha": 0.05,
    "n_cal": 10,
    "n_test": 10,
    "provenance": {"supported_distribution": "SIMULATOR-DOMAIN ONLY: test fixture"},
    "genes": {},
}

_SOURCE_MODEL_SHA256 = "deadbeef" * 8   # fake-but-64-hex-char .pt sha256, for tests only


def _card_extra(**overrides):
    card = {"model_id": "alignair-aa-igh-human", "model_version": "0.0.0-test",
            "calibration_scope": "TEST §8 SCOPE STRING", "license": "GPL-3.0-or-later",
            "citation": "AlignAIR", "source_model_sha256": _SOURCE_MODEL_SHA256}
    card.update(overrides)
    return card


def _build_bundle(tmp_path, filename="bundle.alignair", **card_overrides):
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    dc = ga.HUMAN_IGH_OGRDB
    asc_multi = asc_to_imgt_multi(dc)
    d_allele_order = [name for name, _ in sorted(meta["d_index"].items(), key=lambda kv: kv[1])]
    path = tmp_path / filename
    manifest = save_aa_bundle(
        str(path), state_dict=model.state_dict(), cfg=cfg, class_maps=class_maps,
        d_allele_order=d_allele_order, asc_multi=asc_multi, calibration=_CALIBRATION,
        card_extra=_card_extra(**card_overrides))
    return path, manifest


def test_canonical_json_is_sorted_compact_and_deterministic():
    a = canonical_json({"b": 1, "a": 2, "z": "x"})
    b = canonical_json({"z": "x", "a": 2, "b": 1})
    assert a == b
    assert a == b'{"a":2,"b":1,"z":"x"}'


def test_bundle_is_alignair_file_with_clean_card(tmp_path):
    path, manifest = _build_bundle(tmp_path)

    assert container.is_alignair_file(str(path))

    header = container.read_header(str(path))
    assert header["artifact_type"] == "alignair-aa-model"
    assert "host" not in header
    assert "user" not in header
    assert header["bundle_schema_version"] == "alignair.aa.bundle.v1"
    assert header["model_id"] == "alignair-aa-igh-human"
    assert header["model_version"] == "0.0.0-test"
    assert header["calibration_scope"] == "TEST §8 SCOPE STRING"
    assert header["max_seq_length"] == 192
    assert set(["weights", "config", "tokenizer", "class_maps", "asc_imgt", "calibration"]) \
        <= set(header["sections"])

    # the returned manifest reflects exactly what was written to disk
    assert manifest["artifact_type"] == "alignair-aa-model"
    assert "host" not in manifest
    assert "user" not in manifest


def test_class_map_sha256_is_stable_across_two_builds_of_same_inputs(tmp_path):
    _, m1 = _build_bundle(tmp_path, filename="b1.alignair")
    _, m2 = _build_bundle(tmp_path, filename="b2.alignair")

    h1 = m1["reference_provenance"]["class_map_sha256"]
    h2 = m2["reference_provenance"]["class_map_sha256"]
    assert h1 == h2
    assert h1 != ""

    # ditto for the other canonical-JSON-hashed fields (asc table, tokenizer are content-fixed)
    assert m1["reference_provenance"]["asc_table_sha256"] == m2["reference_provenance"]["asc_table_sha256"]
    assert m1["reference_provenance"]["tokenizer_sha256"] == m2["reference_provenance"]["tokenizer_sha256"]


def test_calibration_provenance_pulled_from_top_level_and_provenance_block(tmp_path):
    _, manifest = _build_bundle(tmp_path)
    cp = manifest["calibration_provenance"]
    assert cp["alpha"] == 0.05
    assert cp["n_cal"] == 10
    assert cp["n_test"] == 10
    assert cp["supported_distribution"] == "SIMULATOR-DOMAIN ONLY: test fixture"
    # method/lambda/k_reg are absent from this calibration shape, so they fill to the
    # FrozenConformal defaults (APS, lambda=0.0, k_reg=1) rather than staying null (Task-3 §4 binding).
    assert cp["method"] == "APS"
    assert cp["lambda"] == 0.0
    assert cp["k_reg"] == 1


def test_reference_provenance_carries_source_model_sha256(tmp_path):
    _, manifest = _build_bundle(tmp_path)
    assert manifest["reference_provenance"]["source_model_sha256"] == _SOURCE_MODEL_SHA256


def test_embedded_calibration_is_complete_with_method_block(tmp_path):
    path, _ = _build_bundle(tmp_path)
    calibration = json.loads(container.read_section(str(path), "calibration"))
    method = calibration["method"]
    assert method["name"] == "APS"
    assert method["lambda"] == 0.0
    assert method["k_reg"] == 1
    assert method["alpha"] == 0.05


def test_complete_calibration_never_overwrites_explicit_method_fields():
    calibration = {"alpha": 0.1, "method": {"name": "RAPS", "alpha": 0.1, "lambda": 0.3, "k_reg": 2}}
    out = complete_calibration(calibration)
    assert out["method"] == {"name": "RAPS", "alpha": 0.1, "lambda": 0.3, "k_reg": 2}
    # and the original dict is untouched (a genuine copy, not an in-place mutation)
    assert calibration["method"] == {"name": "RAPS", "alpha": 0.1, "lambda": 0.3, "k_reg": 2}


def test_complete_calibration_fills_missing_alpha_from_top_level():
    out = complete_calibration({"alpha": 0.07})
    assert out["method"]["alpha"] == 0.07
    assert out["method"]["name"] == "APS"
    assert out["method"]["lambda"] == 0.0
    assert out["method"]["k_reg"] == 1


def test_integrity_hashes_match_recomputation(tmp_path):
    path, manifest = _build_bundle(tmp_path)
    weights_bytes = container.read_section(str(path), "weights")
    config_bytes = container.read_section(str(path), "config")
    calibration_bytes = container.read_section(str(path), "calibration")
    assert manifest["integrity"]["weights_sha256"] == container.sha256_hex(weights_bytes)
    assert manifest["integrity"]["config_sha256"] == container.sha256_hex(config_bytes)
    assert manifest["integrity"]["calibration_sha256"] == container.sha256_hex(calibration_bytes)
