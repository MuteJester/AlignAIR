"""Publish an artifact into a (local) registry directory — the maintainer-side write path.

**Transactional** (P0-11): the artifact is *staged* and the new ``registry.json`` is built in memory,
then the validator runs against that staged state. Only if it passes are the artifact and catalog
committed atomically (rename); on failure nothing is written — no updated catalog and no copied invalid
artifact are left behind. HF upload is a maintainer follow-on (``Aligner.push_to_hub``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..model_file import read_metadata
from .cache import _lock, _version_key
from .validate import validate_registry


def publish_local(artifact_path: str, model_id: str, version: str, registry_dir: str, *,
                  description: str | None = None, entry_extra: dict | None = None) -> list[str]:
    registry_dir = Path(registry_dir)
    data = Path(artifact_path).read_bytes()
    md = read_metadata(artifact_path)
    rel = f"{model_id}/{version}.alignair"
    (registry_dir / model_id).mkdir(parents=True, exist_ok=True)
    final = registry_dir / rel
    # per-registry lock: serialize concurrent publishers so they can't clobber each other's staged/tmp
    # files or lose each other's registry updates (audit #9). Process-unique staging paths as well.
    with _lock(registry_dir / "registry.json.lock"):
        return _publish_locked(data, md, model_id, version, registry_dir, rel, final,
                               description, entry_extra)


def _publish_locked(data, md, model_id, version, registry_dir, rel, final, description, entry_extra):
    staged = final.with_name(f"{final.name}.staging.{os.getpid()}")
    staged.write_bytes(data)                        # stage; the live artifact at `final` is untouched
    try:
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

        # VALIDATE the staged state before committing anything (the new rel resolves to the staged file)
        problems = validate_registry(reg, lambda r: str(staged) if r == rel else str(registry_dir / r))
        if problems:
            return problems                         # abort: registry.json + `final` left untouched

        os.replace(staged, final)                   # commit artifact (atomic)
        staged = None
        tmp_reg = reg_path.with_name(f"{reg_path.name}.tmp.{os.getpid()}")
        tmp_reg.write_text(json.dumps(reg, indent=2, sort_keys=True))
        os.replace(tmp_reg, reg_path)               # commit catalog (atomic)
        return []
    finally:
        if staged is not None and staged.exists():  # validation failed or raised -> discard the stage
            try:
                staged.unlink()
            except OSError:
                pass
