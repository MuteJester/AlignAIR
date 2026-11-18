"""Architectures registered outside the package: a model whose architecture is not registered is resolved by importing
the modules ``ALIGNAIR_ARCHITECTURE_MODULES`` names (comma-separated), which register it; any process that loads the
model (a subprocess included) takes them from its environment."""
import sys

import pytest

from alignair.core import components
from alignair.core.components import ARCHITECTURE_MODULES, architecture_components
from alignair.errors import ConfigurationError

pytestmark = pytest.mark.component

PLUGIN = '''
from alignair.core import register_architecture
from alignair.core.components import CHIP
register_architecture("test-plugin-v1", CHIP)
'''


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    (tmp_path / "test_alignair_plugin.py").write_text(PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(components, "_ARCHITECTURES", dict(components._ARCHITECTURES))
    yield "test_alignair_plugin"
    sys.modules.pop("test_alignair_plugin", None)


def test_an_unregistered_architecture_is_registered_by_the_named_modules(plugin, monkeypatch):
    monkeypatch.setenv(ARCHITECTURE_MODULES, f" {plugin} ,")
    assert isinstance(architecture_components("test-plugin-v1"), components.ModelComponents)


def test_a_named_module_that_cannot_be_imported_is_reported(plugin, monkeypatch):
    monkeypatch.setenv(ARCHITECTURE_MODULES, "no_such_alignair_plugin_module")
    with pytest.raises(ConfigurationError, match="no_such_alignair_plugin_module"):
        architecture_components("test-plugin-v1")


def test_without_the_variable_an_unregistered_architecture_is_refused(plugin, monkeypatch):
    monkeypatch.delenv(ARCHITECTURE_MODULES, raising=False)
    with pytest.raises(ConfigurationError, match="unknown model architecture") as caught:
        architecture_components("test-plugin-v1")
    assert ARCHITECTURE_MODULES in caught.value.hint
