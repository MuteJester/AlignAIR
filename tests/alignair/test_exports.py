"""What the package exports (P2-D10). Every error class lives in and is exported by ``alignair.errors``."""
import importlib
import pkgutil

import pytest

import alignair

pytestmark = pytest.mark.component


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
