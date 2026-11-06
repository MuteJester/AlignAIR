"""A model file carries its reference and needs no pickle: loading takes neither dataconfigs nor trust_pickle, and the
files written before this change load and predict as they did."""
import contextlib
import dataclasses
import inspect
import io
import json

import pytest

from alignair import api
from alignair.aligner import Aligner
from alignair.cli.main import main
from alignair.train.config import TrainingConfig
from tests.alignair.predict.golden_rows import FIXTURE, TEST_CHIP, predict_rows

pytestmark = pytest.mark.component


def _help(verb: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        main([verb, "--help"])
    return out.getvalue()


def test_loading_takes_no_dataconfigs_or_trust_pickle():
    assert set(inspect.signature(api.load_model).parameters) == {"checkpoint_path", "reference", "device"}
    assert {"dataconfigs", "trust_pickle"}.isdisjoint(inspect.signature(Aligner.from_pretrained).parameters)
    assert "trust_pickle" not in {field.name for field in dataclasses.fields(TrainingConfig)}
    assert "--dataconfig" not in _help("predict") and "--trust-pickle" not in _help("predict")
    assert "--trust-pickle" not in _help("benchmark")


def test_an_existing_model_file_loads_and_predicts_its_rows():
    fixture = json.loads(FIXTURE.read_text())
    aligner = Aligner.from_pretrained(str(TEST_CHIP), device="cpu")
    assert predict_rows(aligner.model.eval(), aligner.reference, fixture["reads"]) == fixture["rows"]
