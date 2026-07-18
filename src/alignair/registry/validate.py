"""Registry validator — the gate every published artifact must pass (run in CI / by publish).

Checks each ``registry.json`` version against its artifact: size + SHA256 match; the artifact is
**pickle-free** (no ``dataconfig/*`` / ``train_state``) and carries ``reference_json``; the card's
``model_id``/``model_version`` match the registry; the reference hashes recompute from the embedded
``reference_json``; and versions are unique + ``latest`` resolves. Returns a list of problems ([]==ok).
"""
from __future__ import annotations

import hashlib
import os

from ..model_file import container, read_metadata, serialize

REQUIRED_CARD_FIELDS = ("model_id", "model_version", "model_format_version", "created_by_alignair")


def validate_registry(registry: dict, artifact_path) -> list[str]:
    """``artifact_path(relpath)`` -> local path of an artifact. Returns problem strings."""
    problems: list[str] = []
    for mid, m in registry.get("models", {}).items():
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
    data = open(path, "rb").read()
    if entry.get("size") is not None and entry["size"] != len(data):
        problems.append(f"{mid}@{ver}: size {len(data)} != registry {entry['size']}")
    if entry.get("artifact_sha256") and entry["artifact_sha256"] != hashlib.sha256(data).hexdigest():
        problems.append(f"{mid}@{ver}: artifact_sha256 mismatch")

    md = read_metadata(path)
    secs = md.get("sections", {})
    if any(k.startswith("dataconfig/") or k == "train_state" for k in secs):
        problems.append(f"{mid}@{ver}: contains a pickle section (dataconfig/train_state) — not distributable")
    if "reference_json" not in secs:
        problems.append(f"{mid}@{ver}: missing safe reference_json section")
    else:                                                          # reference hashes must recompute
        ref = serialize.reference_from_json(container.read_section(path, "reference_json"))
        if serialize.allele_order_sha256(ref) != md.get("reference", {}).get("allele_order_sha256"):
            problems.append(f"{mid}@{ver}: allele_order_sha256 does not match its reference_json")
    for f in REQUIRED_CARD_FIELDS:
        if not md.get(f):
            problems.append(f"{mid}@{ver}: card missing required field '{f}'")
    if md.get("model_id") not in (None, mid):
        problems.append(f"{mid}@{ver}: card model_id '{md.get('model_id')}' != registry id '{mid}'")
    if md.get("model_version") not in (None, ver):
        problems.append(f"{mid}@{ver}: card model_version '{md.get('model_version')}' != registry '{ver}'")
    for hk in ("allele_order_sha256", "reference_fasta_sha256"):
        if entry.get(hk) and md.get("reference", {}).get(hk) != entry[hk]:
            problems.append(f"{mid}@{ver}: {hk} in registry != card")
