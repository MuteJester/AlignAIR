"""A run directory's manifest, shared by ``all`` (``workflow.orchestrate``) and ``real`` (``workflow.real``).

The manifest records the configuration a run's outputs depend on (its ``identity``), hashed as its ``config_hash``
(``core.records.digest``), with the host, the code's git state and the GenAIRR in use (``manifest_record``). A run
directory made for another configuration is refused before anything changes, naming the fields that differ
(``changed_fields`` over ``flat``, described by ``describe_changes``; a field one identity lacks, as in a manifest
written before the field existed, is shown ``(absent)``), unless forced.
"""
from __future__ import annotations

import os
import platform
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.provenance import genairr_provenance, git_state
from alignair_assay.runners.variants import parse_tool

# A field one of two identities lacks.
ABSENT = object()


def tool_order_warnings(tools: Sequence[str]) -> list[str]:
    """What may be read backwards or mixed in a comparison of the checkpoint tools of ``tools`` (in the order given,
    where Δ = the second − the first): a tool labelled ``base`` after another tool of its base, and two variants of
    different bases (``alignair-local@x`` with ``alignair-api@y``: the CLI and the API, two pipelines). Kept out of
    ``runners.variants``, whose source is part of a variant's real run identity."""
    warnings = []
    seen: list[str] = []
    for tool in tools:
        base, label = parse_tool(tool)
        earlier = next((other for other in seen if parse_tool(other)[0] == base), None)
        if label == "base" and earlier is not None:
            warnings.append(f"{tool} comes after {earlier}, so Δ = {tool} − {earlier}: give the baseline first "
                            f"(--tools {tool},{earlier})")
        seen.append(tool)
    variants = [tool for tool in tools if parse_tool(tool)[1] is not None]
    mixed = next(((a, b) for i, a in enumerate(variants) for b in variants[i + 1:]
                  if parse_tool(a)[0] != parse_tool(b)[0]), None)
    if mixed is not None:
        warnings.append(f"{mixed[0]} and {mixed[1]} run different pipelines (the CLI and the Python API): their Δ "
                        "holds the pipeline as well as the checkpoint; compare variants of one base")
    return warnings


def flat(identity: dict) -> dict:
    """``identity`` with each tool's setting as its own ``<tool>.<key>`` field (``settings``) and each image as
    ``images.<name>`` (``images``), for naming what differs."""
    fields = {name: value for name, value in identity.items() if name not in ("settings", "images")}
    for tool, values in (identity.get("settings") or {}).items():
        fields.update({f"{tool}.{key}": value for key, value in values.items()})
    fields.update({f"images.{name}": image for name, image in (identity.get("images") or {}).items()})
    return fields


def changed_fields(previous: dict, current: dict, *, ignore: Iterable[str] = ()) -> list[str]:
    """The ``flat`` fields that differ between identities ``previous`` and ``current``, sorted, the top-level keys
    ``ignore`` left out of both; a field one lacks differs."""
    ignore = set(ignore)
    before = flat({key: value for key, value in (previous or {}).items() if key not in ignore})
    after = flat({key: value for key, value in current.items() if key not in ignore})
    return [name for name in sorted(before.keys() | after.keys())
            if before.get(name, ABSENT) != after.get(name, ABSENT)]


def describe_changes(previous: dict, current: dict, changed: Iterable[str]) -> str:
    """``<field> <before> -> <after>`` for each of ``changed``, joined by ``; ``; a field one lacks is
    ``(absent)``."""
    before, after = flat(previous or {}), flat(current)
    return "; ".join(f"{name} {_shown(before.get(name, ABSENT))} -> {_shown(after.get(name, ABSENT))}"
                     for name in changed)


def manifest_record(*, kind: str, version: str, config: dict, identity: dict, config_hash: str, split: str,
                    replaced: dict | None, **extra) -> dict:
    """The manifest: ``kind``, ``version``, ``config``, ``identity``, ``config_hash`` and ``split``; then the ``host``
    (name, platform, Python, logical CPUs), the code's ``git`` state, the ``genairr`` in use, ``created_utc`` and
    ``replaced`` (the manifest it replaced: its hash and the fields that changed; None); then ``extra``."""
    return {"kind": kind, "version": version, "config": config, "identity": identity, "config_hash": config_hash,
            "split": split,
            "host": {"hostname": platform.node(), "platform": platform.platform(),
                     "python": platform.python_version(), "logical_cpus": os.cpu_count()},
            "git": git_state(Path(__file__).resolve().parent), "genairr": genairr_provenance(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "replaced": replaced, **extra}


def _shown(value) -> str:
    return "(absent)" if value is ABSENT else str(value)
