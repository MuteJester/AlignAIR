"""Tests for `AAPredictor.predict` (Task 5): calls + §5 coordinates + junction_aa, no calibration yet."""
from __future__ import annotations

import torch

from alignair.aa.equivalence import EquivalenceClassMap
from alignair.aa.product.loader import AAPredictor, load_aa_model
from alignair.aa.product.schema import AAPrediction
from alignair.aa.product.validate import ErrorResult

_REAL_MODEL_ID = "alignair-aa-igh-human@2.0.0"

# Two real, full-length VH amino-acid reads (from docs/superpowers/artifacts/2026-07-21-aa-phase3/
# hydrated_sequences.tsv — real antibody VH sequences used elsewhere in this repo's AA test suite).
_VH_1 = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAV"
        "YYCSRWGGDGFYAMDYWGQGTLVTVSS")
_VH_2 = ("EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAV"
        "YYCAKYPHYYGSSHWYFDVWGQGTLVTVSS")


# --------------------------------------------------------------------------------- real v2 bundle


def test_predict_real_v2_bundle_returns_full_schema_for_two_vh_reads():
    predictor = load_aa_model(_REAL_MODEL_ID)
    results = predictor.predict([_VH_1, _VH_2])

    assert len(results) == 2
    for seq, result in zip((_VH_1, _VH_2), results):
        assert isinstance(result, AAPrediction)
        assert result.input_length == len(seq)
        assert set(result.segments) == {"V", "D", "J"}

        for gene in ("V", "D", "J"):
            seg = result.segments[gene]
            top_call = seg["top_call"]
            assert isinstance(top_call["genes"], list) and len(top_call["genes"]) >= 1
            assert isinstance(top_call["families"], list) and len(top_call["families"]) >= 1
            assert isinstance(top_call["top_alleles"], list) and len(top_call["top_alleles"]) >= 1
            assert 0.0 <= top_call["probability"] <= 1.0

            coords = seg["coordinates"]
            if coords is not None:
                assert isinstance(coords["start"], int) and isinstance(coords["end"], int)
                assert 0 <= coords["start"] <= len(seq)
                assert 0 <= coords["end"] <= len(seq)
                assert coords["start"] <= coords["end"]

            # Task-7 placeholders — empty/zero until calibration is wired in
            assert seg["calibrated_class_set"] == []
            assert seg["calibrated_allele_set"] == []
            assert seg["calibrated_gene_set"] == []
            assert seg["calibrated_family_set"] == []
            assert seg["set_size"] == {"classes": 0, "alleles": 0, "genes": 0}

        # D segment documents allele-head-index semantics: class_id indexes d_allele_order directly
        d_call = result.segments["D"]["top_call"]
        assert predictor.d_allele_order[d_call["class_id"]] == d_call["top_alleles"][0]
        assert d_call["class_name"] == d_call["top_alleles"][0]

        assert result.applied_calibration_group == {"group": "pooled", "source": "pooled_fallback"}

        prov = result.provenance
        assert prov["model_id"] == "alignair-aa-igh-human"
        assert prov["model_version"] == "2.0.0"
        assert prov["weights_sha256"]
        assert prov["calibration_sha256"]
        assert prov["class_map_db_hash"]
        assert prov["alignair_version"]
        assert "simulator" in prov["calibration_scope"].lower()

        if result.junction_aa is not None:
            assert isinstance(result.junction_aa, str)

        d = result.to_dict()
        assert d["sequence_id"] == result.sequence_id
        assert d["segments"]["V"]["top_call"]["probability"] == result.segments["V"]["top_call"]["probability"]


def test_predict_generates_sequential_ids_when_not_supplied():
    predictor = load_aa_model(_REAL_MODEL_ID)
    results = predictor.predict([_VH_1, _VH_2])
    assert [r.sequence_id for r in results] == ["seq_0", "seq_1"]


def test_predict_uses_supplied_sequence_id():
    predictor = load_aa_model(_REAL_MODEL_ID)
    [result] = predictor.predict([{"sequence_id": "my-read", "sequence": _VH_1}])
    assert result.sequence_id == "my-read"


def test_predict_on_invalid_skip_interleaves_errors_in_original_order():
    predictor = load_aa_model(_REAL_MODEL_ID)
    results = predictor.predict([_VH_1, "EV*QL", _VH_2], on_invalid="skip")
    assert len(results) == 3
    assert isinstance(results[0], AAPrediction) and results[0].sequence_id == "seq_0"
    assert isinstance(results[1], ErrorResult) and results[1].sequence_id == "seq_1"
    assert results[1].error == "invalid_character"
    assert isinstance(results[2], AAPrediction) and results[2].sequence_id == "seq_2"


def test_predict_on_invalid_error_raises_on_first_bad_record():
    predictor = load_aa_model(_REAL_MODEL_ID)
    try:
        predictor.predict([_VH_1, "EV*QL"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "seq_1" in str(exc)


# --------------------------------------------------------------------------------- crafted start>end coordinates


def _minimal_class_map(gene: str, allele: str) -> EquivalenceClassMap:
    return EquivalenceClassMap(gene=gene, db_hash="fakehash",
                               allele_class={allele: 0}, class_members={0: [allele]})


class _StubModel:
    """Ignores its input and always returns a fixed forward-pass output — used to force a
    deterministic V start>end without needing real model weights."""

    def __init__(self, out: dict):
        self._out = out

    def eval(self):
        return self

    def __call__(self, batch):
        return self._out


class _StubTokenizer:
    def encode(self, sequence: str):
        import numpy as np
        return np.ones(len(sequence), dtype=np.int64)


class _StubCfg:
    max_seq_length = 32
    has_d = True


def _fake_predictor() -> AAPredictor:
    out = {
        "v_class_logits": torch.tensor([[2.0]]),
        "j_class_logits": torch.tensor([[2.0]]),
        "d_class_logits": torch.tensor([[2.0]]),
        # V: start(8) > end(3) after rounding/clipping -> must be voided
        "v_start": torch.tensor([[8.0]]), "v_end": torch.tensor([[3.0]]),
        # J and D stay coherent and inside [0, 10]
        "j_start": torch.tensor([[5.0]]), "j_end": torch.tensor([[9.0]]),
        "d_start": torch.tensor([[4.0]]), "d_end": torch.tensor([[6.0]]),
    }
    return AAPredictor(
        model=_StubModel(out),
        cfg=_StubCfg(),
        tokenizer=_StubTokenizer(),
        class_maps={"V": _minimal_class_map("V", "IGHV1-1*01"),
                   "J": _minimal_class_map("J", "IGHJ1*01")},
        d_allele_order=["IGHD1-1*01"],
        asc_multi={"IGHV1-1*01": ["IGHV1-1*01"], "IGHJ1*01": ["IGHJ1*01"]},
        conformal={},
        card={"model_id": "fake-model", "model_version": "0.0.0-fake",
             "alignair_version": "3.0.0-test", "calibration_scope": "TEST SIMULATOR SCOPE STRING",
             "integrity": {"weights_sha256": "abc123", "calibration_sha256": "def456"},
             "reference_provenance": {"class_map_db_hash": {"V": "vhash", "J": "jhash"}}},
        device="cpu",
    )


def test_predict_voids_coordinates_and_warns_on_start_greater_than_end():
    predictor = _fake_predictor()
    [result] = predictor.predict(["ACDEFGHIKL"])            # length 10

    assert result.segments["V"]["coordinates"] is None
    assert "V: incoherent start>end" in result.warnings

    # J and D were coherent and are untouched
    assert result.segments["J"]["coordinates"] == {"start": 5, "end": 9}
    assert result.segments["D"]["coordinates"] == {"start": 4, "end": 6}

    # junction_aa cannot be derived because V's coordinates were voided
    assert result.junction_aa is None


def test_predict_cross_gene_overlap_warns_without_repair():
    predictor = _fake_predictor()
    predictor.model._out["v_start"] = torch.tensor([[0.0]])
    predictor.model._out["v_end"] = torch.tensor([[7.0]])     # V end (7) > J start (5) -> overlap
    predictor.model._out["j_start"] = torch.tensor([[5.0]])
    predictor.model._out["j_end"] = torch.tensor([[9.0]])

    [result] = predictor.predict(["ACDEFGHIKL"])

    assert result.segments["V"]["coordinates"] == {"start": 0, "end": 7}
    assert result.segments["J"]["coordinates"] == {"start": 5, "end": 9}
    assert "segment coordinates overlap/out-of-order" in result.warnings
    # not repaired: coordinates are reported exactly as produced
    assert result.junction_aa is None


def test_predict_junction_aa_is_the_v_end_to_j_start_substring():
    predictor = _fake_predictor()
    predictor.model._out["v_start"] = torch.tensor([[0.0]])
    predictor.model._out["v_end"] = torch.tensor([[4.0]])
    predictor.model._out["j_start"] = torch.tensor([[7.0]])
    predictor.model._out["j_end"] = torch.tensor([[10.0]])

    seq = "ACDEFGHIKL"
    [result] = predictor.predict([seq])

    assert result.segments["V"]["coordinates"] == {"start": 0, "end": 4}
    assert result.segments["J"]["coordinates"] == {"start": 7, "end": 10}
    assert result.junction_aa == seq[4:7]
