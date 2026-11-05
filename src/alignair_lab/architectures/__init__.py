"""Candidate architectures: importing this package registers every candidate with ``alignair.core.components``.

One module per candidate family, each calling ``register`` for every version it defines when imported; this package
imports each of its modules (those whose names do not start with ``_``). A model file names its architecture and
loading rebuilds that name's modules, so the factories registered under a name never change: a change to a candidate
that changes its weights or its behaviour is a new version, ``<family>-v<N+1>``, registered beside the old one.

A process that builds or loads a candidate's model imports this package first: ``screen`` does, before it trains
and before it diagnoses. Data workers never build a model and need not. Candidate code is product code in shape: it
imports ``alignair``, never the assay or the rest of the lab, so a candidate that wins can move into the product.
"""
from __future__ import annotations

import importlib
import pkgutil

from alignair.core.components import ModelComponents, register_architecture
from alignair.errors import ConfigurationError

# A candidate's name: lower-case words joined by hyphens, ending in its version, e.g. ``lab-null-v2``.
NAME = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*-v[0-9]+"
_REGISTERED: list[str] = []


def register(name: str, components: ModelComponents) -> None:
    """Register a candidate version under ``name`` (``NAME``), as ``alignair.core.components.register_architecture``
    does: an existing name is refused."""
    import re
    if not isinstance(name, str) or not re.fullmatch(NAME, name):
        raise ConfigurationError(f"a candidate architecture needs a versioned name such as 'lab-null-v2', got {name!r}",
                                 context={"field": "architecture"})
    register_architecture(name, components)
    _REGISTERED.append(name)


def registered() -> tuple[str, ...]:
    """The names this package registered, in registration order."""
    return tuple(_REGISTERED)


for _module in sorted(module.name for module in pkgutil.iter_modules(__path__)):
    if not _module.startswith("_"):
        importlib.import_module(f"{__name__}.{_module}")
