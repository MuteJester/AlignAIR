"""P0-9: the stable object API — Aligner / PredictionResult / TrainingConfig, device resolution, and a
public-surface snapshot. Model-backed tests use the shipped IGH model when present."""
import csv
import os

import pytest

import alignair
from alignair import Aligner, PredictionResult, TrainingConfig, resolve_device

_IGH = ".private/models/alignair_igh_v1.cal.alignair"
_have_model = os.path.exists(_IGH)


def test_resolve_device_auto_and_fallback():
    dev = resolve_device("auto")
    assert dev in ("cpu", "cuda", "mps")
    assert resolve_device("cpu") == "cpu"
    # an explicit unavailable backend falls back to cpu (this box has CUDA hidden in tests)
    import torch
    if not torch.cuda.is_available():
        assert resolve_device("cuda") == "cpu"


def test_training_config_presets_and_immutability():
    c = TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="quick")
    assert c.dataconfigs == ("HUMAN_IGH_OGRDB",) and c.steps == 300 and c.batch_size == 16
    assert TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="quick", steps=5).steps == 5  # override
    with pytest.raises(ValueError, match="unknown preset"):
        TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="nope")
    with pytest.raises(Exception):                          # frozen dataclass
        c.steps = 1
    assert c.with_(steps=7).steps == 7 and c.steps == 300   # with_ is non-mutating


def test_public_surface_snapshot():
    for name in ("Aligner", "PredictionResult", "TrainingConfig", "TrainingRun", "train",
                 "resolve_device", "load_model", "predict_sequences", "train_model"):
        assert name in alignair.__all__ and hasattr(alignair, name)
    for m in ("from_pretrained", "from_model", "predict", "predict_iter", "loci", "default_locus"):
        assert hasattr(Aligner, m)


def test_prediction_result_io(tmp_path):
    recs = [{"sequence": "ACGT", "v_call": "IGHV1-1*01", "v_sequence_start": 0, "v_sequence_end": 4},
            {"sequence": "TTTT", "v_call": "IGHV1-2*01", "v_sequence_start": 0, "v_sequence_end": 4}]
    r = PredictionResult(recs, locus="IGH")
    assert len(r) == 2 and [x["v_call"] for x in r] == ["IGHV1-1*01", "IGHV1-2*01"]
    out = r.write_airr(str(tmp_path / "o.tsv"))
    with open(out) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows[0]["v_call"] == "IGHV1-1*01" and rows[0]["locus"] == "IGH"


@pytest.mark.skipif(not _have_model, reason="shipped IGH model not present")
def test_aligner_predict_parity_with_functional_api():
    from alignair.api import load_model, predict_sequences
    seqs = ["CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCT",
            "GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCC"]
    aligner = Aligner.from_pretrained(_IGH, device="cpu")
    result = aligner.predict(seqs, batch_size="auto")
    assert isinstance(result, PredictionResult) and len(result) == 2
    assert result.locus == "IGH"                           # single-locus model labels itself
    model, ref = load_model(_IGH, device="cpu")
    baseline = predict_sequences(model, ref, seqs, device="cpu")
    assert [r["v_call"] for r in result] == [r["v_call"] for r in baseline]   # identical path


@pytest.mark.slow
def test_train_api_end_to_end(tmp_path):
    """TrainingConfig -> train -> TrainingRun.best_aligner -> predict, no private imports."""
    from alignair import Aligner, TrainingConfig, train
    cfg = TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="quick", steps=4, batch_size=4,
                                      val_every=2, grad_clip=1.0)
    run = train(cfg, output_dir=str(tmp_path / "run"))
    assert os.path.exists(run.model_path)
    aligner = run.best_aligner(device="cpu")
    assert isinstance(aligner, Aligner)
    result = aligner.predict(["CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCC"])
    assert len(result) == 1 and result.records[0].get("v_call") is not None
