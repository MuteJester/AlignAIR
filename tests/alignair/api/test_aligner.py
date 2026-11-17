"""The stable object API — Aligner / PredictionResult / TrainingConfig, device resolution, and a
public-surface snapshot. Model-backed tests use the shipped IGH model when present."""
import csv
import os
from dataclasses import FrozenInstanceError

import pytest

import alignair
from alignair import Aligner, PredictionResult, TrainingConfig
from alignair.aligner import resolve_device

pytestmark = pytest.mark.integration


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
    with pytest.raises(FrozenInstanceError):
        c.steps = 1
    assert c.with_(steps=7).steps == 7 and c.steps == 300   # with_ is non-mutating


def test_public_surface_snapshot():
    for name in ("Aligner", "PredictionResult", "TrainingConfig", "TrainingRun", "run_training"):
        assert name in alignair.__all__ and hasattr(alignair, name)
    for m in ("from_pretrained", "from_model", "with_genotype", "predict", "predict_iter", "loci", "default_locus"):
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


def test_aligner_predict_returns_a_result_labelled_with_the_models_locus():
    from pathlib import Path
    igh_model = str(Path(__file__).resolve().parents[1] / "fixtures" / "test_chip.alignair")   # the trained chip
    seqs = ["CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCT",
            "GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCC"]
    aligner = Aligner.from_pretrained(igh_model, device="cpu")
    result = aligner.predict(seqs, batch_size="auto")
    assert isinstance(result, PredictionResult) and len(result) == 2
    assert result.locus == "IGH"                           # single-locus model labels itself
    assert all(record["v_call"] for record in result)


@pytest.mark.slow
def test_train_api_end_to_end(tmp_path):
    """TrainingConfig -> run_training -> TrainingRun.best_aligner -> predict, no private imports."""
    from alignair import Aligner, TrainingConfig, run_training
    cfg = TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="quick", steps=4, batch_size=4,
                                      val_every=2, grad_clip=1.0)
    run = run_training(cfg, output_dir=str(tmp_path / "run"))
    assert os.path.exists(run.model_path)
    aligner = run.best_aligner(device="cpu")
    assert isinstance(aligner, Aligner)
    result = aligner.predict(["CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCC"])
    assert len(result) == 1 and result.records[0].get("v_call") is not None


def _simulated_read(dataconfig_name: str, seed: int = 11) -> str:
    """One GenAIRR read for this locus. TCR loci get ``mutation_rate=0`` (T-cells lack AID, and GenAIRR
    refuses ``mutate()`` on a TCR refdata)."""
    import itertools

    import GenAIRR.data as gd

    from alignair.genairr.experiment import build_experiment
    from alignair.train.records import Curriculum
    params = {**Curriculum().params(0.1), "invert_d_prob": 0.05}
    params["mutation_rate"] = 0.0
    exp = build_experiment(getattr(gd, dataconfig_name), params, allow_curatable=True)
    return next(itertools.islice(exp.stream_records(n=None, seed=seed), 1))["sequence"]


@pytest.mark.slow
def test_tcr_train_api_end_to_end(tmp_path):
    """Training custom models on TCR loci (TRA: non-D, TRB: D-bearing) without AID SHM mutation failures,
    then PREDICTING a simulated read of that locus.

    The models are 2-step (untrained), so accuracy is meaningless here — but a model can only ever emit
    alleles from the reference it is bound to (here its own embedded catalog), so the locus label and the call namespace are
    strong, non-flaky assertions that the whole train -> load -> predict path works per locus."""
    from alignair import Aligner, TrainingConfig, run_training

    # 1. TRA Locus (no D gene)
    cfg_tra = TrainingConfig.from_genairr("HUMAN_TCRA_IMGT", preset="quick", steps=2, batch_size=2)
    run_tra = run_training(cfg_tra, output_dir=str(tmp_path / "run_tra"))
    assert os.path.exists(run_tra.model_path)
    aligner_tra = run_tra.best_aligner(device="cpu")
    assert isinstance(aligner_tra, Aligner)

    res_tra = aligner_tra.predict([_simulated_read("HUMAN_TCRA_IMGT")])
    assert len(res_tra) == 1
    assert res_tra.locus == "TRA"
    rec_tra = res_tra.records[0]
    assert rec_tra["v_call"].startswith("TRAV")          # calls come from THIS locus's catalog
    assert rec_tra["j_call"].startswith("TRAJ")
    assert not rec_tra.get("d_call")                     # TRA has no D segment

    # 2. TRB Locus (has D gene)
    cfg_trb = TrainingConfig.from_genairr("HUMAN_TCRB_IMGT", preset="quick", steps=2, batch_size=2)
    run_trb = run_training(cfg_trb, output_dir=str(tmp_path / "run_trb"))
    assert os.path.exists(run_trb.model_path)
    aligner_trb = run_trb.best_aligner(device="cpu")
    assert isinstance(aligner_trb, Aligner)

    res_trb = aligner_trb.predict([_simulated_read("HUMAN_TCRB_IMGT")])
    assert len(res_trb) == 1
    assert res_trb.locus == "TRB"
    rec_trb = res_trb.records[0]
    assert rec_trb["v_call"].startswith("TRBV")
    assert rec_trb["j_call"].startswith("TRBJ")
    if rec_trb.get("d_call"):                            # D-bearing locus: any D call must be a TRBD
        assert rec_trb["d_call"].startswith("TRBD")


def test_the_label_schema_names_the_alleles_of_the_bound_reference():
    """``label_schema`` gives ``predict_raw``'s allele columns, however the aligner got its reference: a reference
    given at load, or a model wrapped with ``from_model``."""
    import json

    from alignair.genotype import genotype_bank
    from tests.alignair.predict.golden_rows import FIXTURE, TEST_CHIP
    read = json.loads(FIXTURE.read_text())["reads"][0]
    full = Aligner.from_pretrained(str(TEST_CHIP), device="cpu", offline=True)
    subset = genotype_bank(full.reference, {"V": full.reference.gene("V").names[:10]})
    loaded = Aligner.from_pretrained(str(TEST_CHIP), device="cpu", offline=True, reference=subset)
    wrapped = Aligner.from_model(loaded.model, loaded.reference, device="cpu")
    for aligner in (loaded, wrapped):
        assert aligner.label_schema["genes"]["v"] == list(subset.gene("V").names)
        assert len(aligner.label_schema["genes"]["v"]) == aligner.predict_raw([read])["v_allele"].shape[1] == 10
