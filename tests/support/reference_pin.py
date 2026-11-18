"""The byte pin of the references a model trains on: ``PIN`` (``tests/alignair/fixtures/reference_v1_sha256.json``)
holds, for every GenAIRR dataconfig and a few unions, the SHA-256 of its ``alignair.reference.v1`` JSON as a model file
embeds it, and the model card's two reference hashes. ``v1_digest`` measures a reference the same way, leaving out the
optional keys a v1 reader ignores (top-level ``provenance``, each gene's ``info``, each locus's ``species``), so a
moved digest is a moved training target, never a new optional field."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

PIN_PATH = Path(__file__).resolve().parents[1] / "alignair" / "fixtures" / "reference_v1_sha256.json"
PIN = json.loads(PIN_PATH.read_text())


def v1_payload(payload: dict) -> dict:
    """A deep copy of a serialized reference without the optional keys that are not part of the v1 bytes."""
    payload = copy.deepcopy(payload)
    payload.pop("provenance", None)
    for gene in payload.get("genes", {}).values():
        gene.pop("info", None)
    for locus in payload.get("loci", []) or []:
        locus.pop("species", None)
    return payload


def v1_digest(reference) -> str:
    """SHA-256 of ``reference``'s v1 bytes (``model_file.serialize.reference_to_json`` of its v1 payload)."""
    data = json.dumps(v1_payload(reference.to_serializable()), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
