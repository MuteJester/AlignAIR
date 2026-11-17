"""Registry validator — the gate every published artifact must pass (run in CI / by publish).

Checks each ``registry.json`` version against its artifact: size + SHA256 match; the artifact is
**pickle-free** (no ``dataconfig/*`` / ``train_state``) and carries ``reference_json``; the card's
``model_id``/``model_version`` match the registry, and its reference hashes the registry's; both reference hashes
(allele order and FASTA) recompute from the embedded ``reference_json``; every companion file (the card, the
manifest) is present with the registry's SHA256; and ``latest`` names a listed version. Returns a list of problems
([]==ok).
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

from ..io.files import sha256_file
from ..model_file import container, pickle_sections, read_metadata, serialize

REQUIRED_CARD_FIELDS = ("model_id", "model_version", "model_format_version", "created_by_alignair")


def validate_registry(catalog: dict, artifact_path: Callable[[str], str | None]) -> list[str]:
    """The problems of a registry's ``catalog`` (its parsed ``registry.json``, as ``load_registry`` returns it);
    ``artifact_path(relpath)`` is the local path of one of its artifacts. Returns problem strings."""
    problems: list[str] = []
    for mid, m in catalog.get("models", {}).items():
        versions = m.get("versions", {})
        if not versions:
            problems.append(f"{mid}: no versions")
            continue
        if m.get("latest") not in versions:
            problems.append(f"{mid}: latest '{m.get('latest')}' is not a listed version")
        for ver, entry in versions.items():
            _validate_version(problems, mid, ver, entry, artifact_path)
    return problems


def _validate_version(problems, mid, ver, entry, artifact_path) -> None:
    rel = entry.get("file")
    path = artifact_path(rel) if rel else None
    if not path or not os.path.exists(path):
        problems.append(f"{mid}@{ver}: artifact missing ({rel})")
        return
    at = f"{mid}@{ver}"
    problems += _check_bytes(at, entry, path)
    problems += _check_companions(at, entry, artifact_path)
    md = read_metadata(path)
    problems += _check_sections(at, md, path)
    problems += _check_card(at, mid, ver, entry, md)


def _check_bytes(at: str, entry: dict, path) -> list[str]:
    """The artifact's size and SHA256 against the registry's."""
    data = Path(path).read_bytes()
    problems = []
    if entry.get("size") is not None and entry["size"] != len(data):
        problems.append(f"{at}: size {len(data)} != registry {entry['size']}")
    if entry.get("artifact_sha256") and entry["artifact_sha256"] != hashlib.sha256(data).hexdigest():
        problems.append(f"{at}: artifact_sha256 mismatch")
    return problems


def _check_companions(at: str, entry: dict, artifact_path) -> list[str]:
    """Each companion file the registry lists: present, and its SHA256 the registry's."""
    problems = []
    for name, companion in (entry.get("companions") or {}).items():
        cpath = artifact_path(companion.get("file")) if companion.get("file") else None
        if not cpath or not os.path.exists(cpath):
            problems.append(f"{at}: {name} missing ({companion.get('file')})")
        elif companion.get("sha256") and companion["sha256"] != sha256_file(cpath):
            problems.append(f"{at}: {name} sha256 mismatch")
    return problems


def _check_sections(at: str, md: dict, path) -> list[str]:
    """No pickle section, and a reference_json both of the card's reference hashes recompute from."""
    problems = []
    if pickle_sections(md):
        problems.append(f"{at}: contains a pickle section (dataconfig/train_state) — not distributable")
    if "reference_json" not in md.get("sections", {}):
        problems.append(f"{at}: missing safe reference_json section")
        return problems
    ref = serialize.reference_from_json(container.read_section(path, "reference_json"))
    card = md.get("reference", {})
    for key, recompute in (("allele_order_sha256", serialize.allele_order_sha256),
                           ("reference_fasta_sha256", serialize.reference_fasta_sha256)):
        if recompute(ref) != card.get(key):
            problems.append(f"{at}: {key} does not match its reference_json")
    return problems


def _check_card(at: str, mid: str, ver: str, entry: dict, md: dict) -> list[str]:
    """The card's required fields, its id and version against the registry's, and the registry's reference hashes
    against the card's."""
    problems = [f"{at}: card missing required field '{f}'" for f in REQUIRED_CARD_FIELDS if not md.get(f)]
    if md.get("model_id") not in (None, mid):
        problems.append(f"{at}: card model_id '{md.get('model_id')}' != registry id '{mid}'")
    if md.get("model_version") not in (None, ver):
        problems.append(f"{at}: card model_version '{md.get('model_version')}' != registry '{ver}'")
    for hk in ("allele_order_sha256", "reference_fasta_sha256"):
        if entry.get(hk) and md.get("reference", {}).get(hk) != entry[hk]:
            problems.append(f"{at}: {hk} in registry != card")
    return problems
