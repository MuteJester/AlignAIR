"""GenAIRR's dataconfigs: resolved from a name or an object, listed, and named.

A name is one of ``GenAIRR.data.list_configs()`` (matched exactly, else ignoring case); an object is any dataconfig
with receptor metadata. ``name_of`` finds a built-in by content (its checksum against GenAIRR's audit index), so a
copy of a built-in keeps its name and a custom dataconfig has none."""
from __future__ import annotations

import functools
from typing import Iterable

from ..errors import ConfigurationError
from ..reference.reference_set import _locus_of

_HINT = "List them with `alignair reference list`."


def _data():
    import GenAIRR.data as data
    return data


def _text(value) -> str:
    return str(getattr(value, "value", value))


def dataconfig_names() -> tuple[str, ...]:
    """The names of GenAIRR's built-in dataconfigs, without loading any of them."""
    return tuple(_data().list_configs())


def resolve_dataconfig(spec):
    """The dataconfig ``spec`` names (or ``spec`` itself, when it is a dataconfig); ConfigurationError otherwise."""
    if isinstance(spec, str):
        names = dataconfig_names()
        name = spec if spec in names else next((n for n in names if n.lower() == spec.lower()), None)
        if name is not None:
            return getattr(_data(), name)
    elif spec is not None and hasattr(getattr(spec, "metadata", None), "chain_type"):
        return spec
    raise ConfigurationError(f"unknown GenAIRR dataconfig {spec!r}",
                             context={"field": "dataconfigs", "dataconfig": str(spec)}, hint=_HINT)


def resolve_dataconfigs(specs: Iterable) -> list:
    """``resolve_dataconfig`` of each of ``specs``, in order."""
    return [resolve_dataconfig(spec) for spec in specs]


def list_dataconfigs() -> list[dict]:
    """One row per built-in dataconfig: ``name``, ``species``, ``chain_type``, ``locus``, ``has_d`` (loads each)."""
    rows = []
    for name in dataconfig_names():
        metadata = getattr(_data(), name).metadata
        rows.append({"name": name, "chain_type": _text(metadata.chain_type), "species": _text(metadata.species),
                     "locus": _locus_of(metadata.chain_type), "has_d": bool(metadata.has_d)})
    return rows


@functools.lru_cache(maxsize=1)
def _builtin_by_checksum() -> dict[str, str]:
    data = _data()
    return {data.audit_config(name)["dataconfig_checksum"]: name for name in data.list_configs()}


def name_of(dataconfig) -> str | None:
    """The built-in name of ``dataconfig`` (a built-in or a copy of one), or None for a custom dataconfig."""
    checksum = getattr(dataconfig, "compute_checksum", None)
    return None if checksum is None else _builtin_by_checksum().get(checksum())


def dataconfig_name(dataconfig) -> str:
    """A name for ``dataconfig`` in records and messages: its built-in name, else its own name, else its chain type. A
    failure to look up the built-ins (a provider that cannot load) never stops a stream from being named."""
    try:
        builtin = name_of(dataconfig)
        if builtin is not None:
            return builtin
    except Exception:  # noqa: BLE001 - naming must not prevent a source from being built
        pass
    name = getattr(dataconfig, 'name', None)
    if name and str(name).strip().lower() not in ('none', 'unnamed'):
        return str(name)
    chain_type = getattr(getattr(dataconfig, 'metadata', None), 'chain_type', 'unknown')
    return str(getattr(chain_type, 'value', chain_type))


def locus_of(dataconfig) -> str:
    """The AIRR locus code of ``dataconfig``'s chain type (``IGH``, ``TRB``, ...)."""
    return _locus_of(dataconfig.metadata.chain_type)
