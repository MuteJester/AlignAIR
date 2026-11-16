"""Publish an artifact into a (local) registry directory — the maintainer-side write path.

**Transactional**: the artifact is *staged* and the new ``registry.json`` is built in memory,
then the validator runs against that staged state. Only if it passes are the artifact and catalog
committed atomically (rename); on failure nothing is written — no updated catalog and no copied invalid
artifact are left behind.

``publish_bundle`` publishes an exported bundle (``alignair export``) with its **companions**, the model card and the
reference manifest, laid out beside the artifact as the catalog serves them (``<id>/<version>.alignair``,
``<id>/<version>.model_card.md``, ``<id>/<version>.reference_manifest.json``) and recorded in the version's
``registry.json`` entry with their SHA-256. ``upload_version`` then sends one version's files and the catalog to the
shared Hugging Face repo in one commit (maintainer-only: a write token).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..errors import MissingDependencyError, RegistryError
from ..model_file import read_metadata
from .cache import cache_lock, version_key
from .hf import hf_token
from .sources import DEFAULT_REGISTRY
from .validate import validate_registry

# a bundle's companion files, by the name its registry entry records them under
COMPANIONS = {"model_card": "model_card.md", "reference_manifest": "reference_manifest.json"}
DEFAULT_REPO_ID = DEFAULT_REGISTRY[len("hf://"):]           # the shared catalog repo, AlignAIR/AlignAIR-pretrained


def publish_local(artifact_path: str, model_id: str, version: str, registry_dir: str, *,
                  description: str | None = None, entry_extra: dict | None = None,
                  companions: dict | None = None) -> list[str]:
    """Publish ``artifact_path`` as ``model_id@version`` into ``registry_dir``; returns the validator's problems ([]:
    published). ``companions`` ({name: local file}) are staged beside the artifact as ``<id>/<version>.<file name>``,
    validated with it and committed with it."""
    registry_dir = Path(registry_dir)
    data = Path(artifact_path).read_bytes()
    md = read_metadata(artifact_path)
    rel = f"{model_id}/{version}.alignair"
    (registry_dir / model_id).mkdir(parents=True, exist_ok=True)
    final = registry_dir / rel
    extra = {name: (f"{model_id}/{version}.{Path(source).name}", Path(source).read_bytes())
             for name, source in (companions or {}).items()}
    # per-registry lock: serialize concurrent publishers so they can't clobber each other's staged/tmp
    # files or lose each other's registry updates. Process-unique staging paths as well.
    with cache_lock(registry_dir / "registry.json.lock"):
        return _publish_locked(data, md, model_id, version, registry_dir, rel, final,
                               description, entry_extra, extra)


def _publish_locked(data, md, model_id, version, registry_dir, rel, final, description, entry_extra, extra=None):
    staged = final.with_name(f"{final.name}.staging.{os.getpid()}")
    staged.write_bytes(data)                        # stage; the live artifact at `final` is untouched
    pid = os.getpid()
    side = {}                                       # companion rel -> its staged path
    try:
        for crel, payload in (extra or {}).values():
            side[crel] = registry_dir / f"{crel}.staging.{pid}"
            side[crel].write_bytes(payload)
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
        if extra:
            entry["companions"] = {name: {"file": crel, "sha256": hashlib.sha256(payload).hexdigest()}
                                   for name, (crel, payload) in extra.items()}

        reg_path = registry_dir / "registry.json"
        reg = json.loads(reg_path.read_text()) if reg_path.exists() else \
            {"schema": "alignair.registry.v1", "models": {}}
        m = reg["models"].setdefault(model_id, {"versions": {}})
        m["versions"][version] = entry
        m["latest"] = max(m["versions"], key=version_key)
        if description or md.get("description"):
            m.setdefault("description", description or md.get("description"))
        for k in ("species", "receptor", "locus"):
            if md.get(k):
                m.setdefault(k, md[k])

        # VALIDATE the staged state before committing anything (the new rel resolves to the staged file)
        problems = validate_registry(reg, lambda r: str(staged) if r == rel else str(side.get(r, registry_dir / r)))
        if problems:
            return problems                         # abort: registry.json + `final` left untouched

        os.replace(staged, final)                   # commit artifact (atomic)
        staged = None
        for crel in list(side):                     # its companions beside it
            os.replace(side.pop(crel), registry_dir / crel)
        tmp_reg = reg_path.with_name(f"{reg_path.name}.tmp.{os.getpid()}")
        tmp_reg.write_text(json.dumps(reg, indent=2, sort_keys=True))
        os.replace(tmp_reg, reg_path)               # commit catalog (atomic)
        return []
    finally:
        for leftover in [staged, *side.values()]:  # validation failed or raised -> discard the stage
            if leftover is not None and leftover.exists():
                try:
                    leftover.unlink()
                except OSError:
                    pass


def publish_bundle(bundle_dir: str, model_id: str, version: str, registry_dir: str, *,
                   description: str | None = None) -> list[str]:
    """Publish an exported bundle (``alignair export``: ``model.alignair``, ``model_card.md``,
    ``reference_manifest.json``) as ``model_id@version`` into the catalog directory ``registry_dir``, the card and the
    manifest beside the model (``publish_local`` with companions). FileNotFoundError, before anything is written,
    for a bundle missing one of the three files. Returns the validator's problems ([]: published)."""
    bundle = Path(bundle_dir)
    needed = ["model.alignair", *COMPANIONS.values()]
    missing = [name for name in needed if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{bundle_dir} is not a complete bundle: it lacks {', '.join(missing)} "
                                "(alignair export writes all three)")
    return publish_local(str(bundle / "model.alignair"), model_id, version, str(registry_dir), description=description,
                         companions={name: str(bundle / file) for name, file in COMPANIONS.items()})


def upload_version(registry_dir: str, model_id: str, version: str, *, repo_id: str = DEFAULT_REPO_ID,
                   token: str | None = None, revision: str | None = None, api=None) -> list[str]:
    """Maintainer-only: upload ``model_id@version`` from the catalog directory ``registry_dir`` (as ``publish_bundle``
    wrote it) to the shared Hugging Face catalog repo, in one commit: the artifact, its companions and
    ``registry.json``. The local ``registry.json`` becomes the hub's, so build it on the hub's current one. The version
    is validated first (RegistryError when it is missing or fails). ``api`` (an ``HfApi``) defaults to one with
    ``token`` (or ``$HF_TOKEN``). Returns the uploaded paths."""
    registry_dir = Path(registry_dir)
    reg = json.loads((registry_dir / "registry.json").read_text())
    entry = reg.get("models", {}).get(model_id, {}).get("versions", {}).get(version)
    if entry is None:
        raise RegistryError(f"{model_id}@{version} is not in {registry_dir / 'registry.json'}",
                            context={"model_id": model_id, "version": version},
                            hint="Publish it first (publish_bundle).")
    one = {"models": {model_id: {"latest": version, "versions": {version: entry}}}}
    problems = validate_registry(one, lambda r: str(registry_dir / r))
    if problems:
        raise RegistryError(f"{model_id}@{version} fails validation: {'; '.join(problems)}",
                            context={"model_id": model_id, "version": version})
    paths = [entry["file"], *(c["file"] for c in entry.get("companions", {}).values()), "registry.json"]
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as e:                         # pragma: no cover - optional dep
        raise MissingDependencyError("upload_version needs 'huggingface_hub' (`pip install alignair[hub]`)") from e
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token(token))
    operations = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(registry_dir / rel)) for rel in paths]
    api.create_commit(repo_id=repo_id, operations=operations, commit_message=f"{model_id} {version}",
                      repo_type="model", revision=revision)
    return paths
