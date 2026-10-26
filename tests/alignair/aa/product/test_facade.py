"""Tests for the public `alignair.predict_aa` / `alignair.load_aa_model` façade (Task 8, design spec
§9): sequences/records in, `list[dict]` (§6 schema) out, model cached per (model, device)."""
from __future__ import annotations
import pytest

import alignair
from alignair.aa.product.loader import AAPredictor

pytestmark = pytest.mark.component

# A real, full-length VH amino-acid read (same fixture used in test_predictor.py, sourced from
# Notes and Plans/docs/superpowers/artifacts/2026-07-21-aa-phase3/hydrated_sequences.tsv).
_VH_1 = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAV"
        "YYCSRWGGDGFYAMDYWGQGTLVTVSS")
_VH_2 = ("EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAV"
        "YYCAKYPHYYGSSHWYFDVWGQGTLVTVSS")


def test_predict_aa_returns_schema_dicts_for_a_real_vh_read():
    results = alignair.predict_aa([_VH_1])

    assert isinstance(results, list) and len(results) == 1
    d = results[0]
    assert isinstance(d, dict)

    # §6 top-level shape
    assert d["sequence_id"] == "seq_0"
    assert d["input_length"] == len(_VH_1)
    assert set(d["segments"]) == {"V", "D", "J"}
    assert "inter_vj_aa" in d
    assert set(d["applied_calibration_group"]) == {"group", "source"}
    assert "warnings" in d and isinstance(d["warnings"], list)

    prov = d["provenance"]
    for key in ("model_id", "model_version", "weights_sha256", "calibration_sha256",
               "class_map_db_hash", "alignair_version", "calibration_scope"):
        assert key in prov and prov[key], key

    for gene in ("V", "D", "J"):
        seg = d["segments"][gene]
        assert set(seg) == {"top_call", "calibrated_class_set", "calibrated_classes",
                            "calibrated_allele_set", "calibrated_gene_set", "calibrated_family_set",
                            "set_size", "coordinates"}
        top = seg["top_call"]
        assert isinstance(top["genes"], list) and isinstance(top["families"], list)
        assert isinstance(top["top_alleles"], list)

    # never presented as nt AIRR: no CIGAR / productivity / nt-coordinate fields
    for forbidden in ("cigar", "productive", "v_sequence_start", "v_germline_start"):
        assert forbidden not in d


def test_predict_aa_default_model_is_the_v2_igh_human_id():
    [d] = alignair.predict_aa([_VH_1])
    assert d["provenance"]["model_id"] == "alignair-aa-igh-human"
    assert d["provenance"]["model_version"] == "2.0.0"


def test_predict_aa_preserves_input_order_and_sequence_ids():
    results = alignair.predict_aa([{"sequence_id": "read-a", "sequence": _VH_1},
                                   {"sequence_id": "read-b", "sequence": _VH_2}])
    assert [r["sequence_id"] for r in results] == ["read-a", "read-b"]


def test_predict_aa_on_invalid_skip_yields_structured_error_dict():
    results = alignair.predict_aa([_VH_1, "EV*QL"], on_invalid="skip")
    assert len(results) == 2
    assert "provenance" in results[0]
    assert results[1] == {"sequence_id": "seq_1", "error": "invalid_character",
                          "detail": results[1]["detail"]}


def test_predict_aa_caches_the_loaded_predictor_per_model_and_device():
    alignair._AA_PREDICTOR_CACHE.clear()
    alignair.predict_aa([_VH_1])
    assert len(alignair._AA_PREDICTOR_CACHE) == 1
    cached = next(iter(alignair._AA_PREDICTOR_CACHE.values()))
    assert isinstance(cached, AAPredictor)

    alignair.predict_aa([_VH_2])   # same (model, device) key -> reuses the cached predictor, no reload
    assert len(alignair._AA_PREDICTOR_CACHE) == 1
    assert next(iter(alignair._AA_PREDICTOR_CACHE.values())) is cached


def test_load_aa_model_is_reexported_and_matches_the_product_loader():
    from alignair.aa.product.loader import load_aa_model as _direct

    assert alignair.load_aa_model is _direct


def test_predict_aa_string_path_is_not_file_read():
    """§9 contract: a path-like string (e.g., 'reads.fasta' or '/etc/passwd') is treated as a
    sequence, never as a file path. With default on_invalid='error', any string containing invalid
    AA characters (like '/', '.') raises ValueError."""
    import pytest

    # "reads.fasta" contains invalid AA characters ('.', 'f', etc.) so it should raise
    with pytest.raises(ValueError, match="invalid character"):
        alignair.predict_aa("reads.fasta")

    # "/etc/passwd" contains invalid AA characters so it should raise, not read the file
    with pytest.raises(ValueError, match="invalid character"):
        alignair.predict_aa("/etc/passwd")


def test_predict_aa_bare_valid_string_returns_one_result():
    """A single valid AA sequence string is treated as one sequence, not parsed char-by-char."""
    [result] = alignair.predict_aa("EVQLVESGGG")
    assert result["sequence_id"] == "seq_0"
    assert result["input_length"] == 10
    assert "segments" in result


def test_predict_aa_on_invalid_error_default_raises_on_invalid_char():
    """§9.1 documented default: on_invalid='error' causes prediction to raise ValueError on
    encountering an invalid character (not silently skip or wrap in error dict)."""
    import pytest

    # Default on_invalid="error" should raise on the invalid '*'
    with pytest.raises(ValueError, match="stop character"):
        alignair.predict_aa(["EVQLVESGGG", "EV*QL"])
