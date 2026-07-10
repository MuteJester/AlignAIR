"""Publish an artifact into a (local) registry directory — the maintainer-side write path.

Copies the ``.alignair`` under ``<registry_dir>/<id>/<version>.alignair``, computes size + SHA256,
reads the card, upserts the ``registry.json`` version entry (recomputing ``latest``), then runs the
validator and returns any problems (the caller aborts on non-empty). HF upload is a Phase-4 follow-on.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..model_file import read_metadata
from .cache import _version_key
from .validate import validate_registry


def publish_local(artifact_path: str, model_id: str, version: str, registry_dir: str, *,
                  description: str | None = None, entry_extra: dict | None = None) -> list[str]:
    registry_dir = Path(registry_dir)
    data = Path(artifact_path).read_bytes()
    md = read_metadata(artifact_path)
    rel = f"{model_id}/{version}.alignair"
    (registry_dir / model_id).mkdir(parents=True, exist_ok=True)
    (registry_dir / rel).write_bytes(data)

    ref = md.get("reference", {})
    entry = {"file": rel, "artifact_sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
             "allele_order_sha256": ref.get("allele_order_sha256"),
             "reference_fasta_sha256": ref.get("reference_fasta_sha256"),
             "created_by_alignair": md.get("created_by_alignair"),
             "model_format_version": md.get("model_format_version"),
             "min_alignair": md.get("min_alignair"), "trained": md.get("created"),
             "metrics": md.get("metrics")}
    if entry_extra:
        entry.update(entry_extra)

    reg_path = registry_dir / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else \
        {"schema": "alignair.registry.v1", "models": {}}
    m = reg["models"].setdefault(model_id, {"versions": {}})
    m["versions"][version] = entry
    m["latest"] = max(m["versions"], key=_version_key)
    if description or md.get("description"):
        m.setdefault("description", description or md.get("description"))
    for k in ("species", "receptor", "locus"):
        if md.get(k):
            m.setdefault(k, md[k])
    reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True))

    return validate_registry(reg, lambda r: str(registry_dir / r))
