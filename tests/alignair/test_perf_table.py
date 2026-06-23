"""Guard that scripts/perf_table.py keeps importing against the current API (#97).

Importing it resolves alignair.api.predict/LoadedModel, the config/core models, the gym, and
the loss/trainer — so an API rename breaks this fast, without running the (slow) sweep.
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("GenAIRR")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "perf_table.py"


def _load():
    spec = importlib.util.spec_from_file_location("perf_table", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_perf_table_imports_and_exposes_api():
    mod = _load()
    assert set(mod.PRESETS) == {"desktop", "standard"}
    assert callable(mod.make_model) and callable(mod.time_predict) and callable(mod.time_train)
    assert isinstance(mod._cpu_name(), str) and mod._cpu_name()
