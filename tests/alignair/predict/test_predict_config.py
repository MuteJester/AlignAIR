"""Prediction has one genotype mechanism (the genotype is the chip's bank) and one germline reader."""
import dataclasses
import inspect

import pytest

from alignair.predict import PredictConfig, pipeline

pytestmark = pytest.mark.component


def test_prediction_has_one_genotype_mechanism_and_one_reader():
    assert {"germline_reader", "genotype", "genotype_method"}.isdisjoint(
        field.name for field in dataclasses.fields(PredictConfig))
    assert "aligner" not in inspect.signature(pipeline.predict).parameters


@pytest.mark.parametrize("options", [{}, {"threshold": 0.3, "cap": 5, "airr": False}])
def test_for_model_is_the_config_aligner_predict_builds(options, monkeypatch):
    from types import SimpleNamespace

    import torch
    from alignair import aligner
    model = torch.nn.Module()
    model.cfg = SimpleNamespace(max_seq_length=320, has_d=False)
    reference = SimpleNamespace(locus_names=lambda: ("IGK",), default_locus="IGK")
    seen = []
    monkeypatch.setattr(aligner, "_predict", lambda m, s, r, cfg, device: seen.append(cfg) or [])
    aligner.Aligner(model, reference, device="cpu").predict(["ACGT"], batch_size=7, **options)
    assert seen == [PredictConfig.for_model(model, reference, batch_size=7, **options)]
    assert seen[0].loci == ("IGK",)
