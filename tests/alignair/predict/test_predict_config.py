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
