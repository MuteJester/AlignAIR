"""AlignAIR 4 has one prediction surface: no fixed-head contract, no genotype mask and no second germline reader."""
import contextlib
import dataclasses
import importlib.util
import inspect
import io
import tomllib
from pathlib import Path

import pytest

from alignair.cli.main import main
from alignair.core import AlignAIR, ModelComponents
from alignair.predict.threshold import select_alleles

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2]


def test_the_alignair_3_prediction_surface_is_gone():
    assert importlib.util.find_spec("alignair.align") is None
    assert importlib.util.find_spec("alignair.reference.identity") is None
    assert "reference_agnostic" not in {field.name for field in dataclasses.fields(ModelComponents)}
    assert not hasattr(AlignAIR, "reference_agnostic")
    assert "allowed" not in inspect.signature(select_alleles).parameters
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        main(["predict", "--help"])
    assert "--genotype-method" not in out.getvalue()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = [*project["dependencies"], *(d for group in project["optional-dependencies"].values() for d in group)]
    assert not any(name in dependency for dependency in declared for name in ("parasail", "pywfa", "biopython"))
