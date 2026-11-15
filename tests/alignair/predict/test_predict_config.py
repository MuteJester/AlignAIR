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


@pytest.mark.parametrize("overrides", [{}, {"threshold": 0.3, "cap": 5, "airr": False}, {"loci": ("X",)}])
def test_for_model_is_the_config_predict_sequences_builds(overrides, monkeypatch):
    from types import SimpleNamespace
    from alignair import api
    model = SimpleNamespace(cfg=SimpleNamespace(max_seq_length=320, has_d=False))
    reference = SimpleNamespace(locus_names=lambda: ("IGK",))
    seen = []
    monkeypatch.setattr(api, "_predict", lambda m, s, r, cfg, device: seen.append(cfg) or [])
    api.predict_sequences(model, reference, ["ACGT"], device="cpu", batch_size=7, **overrides)
    assert seen == [PredictConfig.for_model(model, reference, batch_size=7, **overrides)]
