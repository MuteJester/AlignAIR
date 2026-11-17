"""What the package exports (P2-D10). The top level is the supported API; every error class lives in and is
exported by ``alignair.errors``."""
import importlib
import pkgutil

import pytest

import alignair

pytestmark = pytest.mark.component

TOP_LEVEL = {"AirrWriter", "AlignAIRError", "Aligner", "PredictionResult", "ReferenceSet", "TrainingConfig",
             "TrainingRun", "__version__", "compare_airr", "iter_sequences", "read_sequences", "run_training",
             "write_airr"}


def test_the_top_level_exports_the_supported_api():
    assert set(alignair.__all__) == TOP_LEVEL
    assert all(hasattr(alignair, name) for name in TOP_LEVEL)


@pytest.mark.parametrize("name, module", [
    ("AlignAIR", "alignair.core"), ("AlignAIRConfig", "alignair.core.config"), ("PredictConfig", "alignair.predict"),
    ("resolve_device", "alignair.aligner")])
def test_an_internal_name_left_the_top_level_and_stays_importable_from_its_module(name, module):
    assert not hasattr(alignair, name)
    assert hasattr(importlib.import_module(module), name)


def _package_error_classes() -> set[type]:
    """Every AlignAIRError subclass the package defines (all of its modules imported)."""
    for module in pkgutil.walk_packages(alignair.__path__, "alignair."):
        importlib.import_module(module.name)
    found, todo = set(), [alignair.AlignAIRError]
    while todo:
        cls = todo.pop()
        if cls.__module__.split(".")[0] == "alignair":
            found.add(cls)
        todo.extend(cls.__subclasses__())
    return found


def test_alignair_errors_defines_and_exports_every_error_class():
    from alignair import errors
    classes = _package_error_classes()
    assert len(classes) > 20
    assert sorted(c.__qualname__ for c in classes if not c.__module__.startswith("alignair.errors.")) == []
    assert sorted(c.__name__ for c in classes if getattr(errors, c.__name__, None) is not c
                  or c.__name__ not in errors.__all__) == []
