"""Versioned, self-describing bundle for a trained DNAlignAIR model.

A bundle is a directory that packages everything needed to deploy one model:
    model.pt          state_dict
    config.json       DNAlignAIRConfig
    reference.json    the default reference, either:
                        {"dataconfigs": [...names...], "locus": "IGH"}        (built-in GenAIRR), or
                        {"genotype": {v/d/j: {name: seq}}, "anchors": {...},
                         "locus": "IGH"}                                       (EMBEDDED — custom/own reference)
    calibration.json  per-gene equivalence-set calibration (optional)
    meta.json         {format_version, notes}
    VERSION
    fingerprint.txt   SHA-256 over the other files (tamper detection)

The module stays free of GenAIRR. For a built-in reference it stores dataconfig NAMES
(the caller rebuilds via GenAIRR). For a custom reference (e.g. trained from FASTA) it
EMBEDS the allele sequences + anchors so the bundle is fully self-contained.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import torch

from ..config.dnalignair_config import DNAlignAIRConfig

DNALIGNAIR_BUNDLE_VERSION = 1
_REQUIRED = ("model.pt", "config.json", "reference.json", "VERSION", "fingerprint.txt")


def compute_fingerprint(bundle_dir) -> str:
    """SHA-256 over every bundle file except fingerprint.txt, in name order."""
    h = hashlib.sha256()
    for p in sorted(Path(bundle_dir).iterdir()):
        if not p.is_file() or p.name == "fingerprint.txt":
            continue
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _embed_reference(reference_set) -> dict:
    """Serialize a ReferenceSet to a JSON-able {genotype, anchors} (names + sequences + anchors)."""
    genotype = {g.lower(): dict(zip(ref.names, ref.sequences))
                for g, ref in reference_set.genes.items()}
    anchors = {g: dict(ref.anchors) for g, ref in reference_set.genes.items() if ref.anchors}
    return {"genotype": genotype, "anchors": anchors or None}


def save_dnalignair_bundle(bundle_dir, *, model, dataconfigs: Optional[Iterable[str]] = None,
                           locus: str = "IGH", reference_set=None,
                           calibration: Optional[dict] = None, notes: Optional[str] = None,
                           training_meta: Optional[dict] = None) -> str:
    """Write a bundle for `model` (a DNAlignAIR with .config and .state_dict()).

    Provide EITHER `dataconfigs` (GenAIRR DataConfig NAMES, for a built-in reference) OR
    `reference_set` (a ReferenceSet whose alleles are EMBEDDED — use this for custom/own
    references with no registered dataconfig name, e.g. models trained from FASTA).

    `training_meta` (seed, preset, steps, lr, ...) is recorded under meta.json["training"] for
    reproducibility, alongside automatic provenance (versions, source commit, content hashes)."""
    import datetime
    from ..provenance import alignair_version, package_versions, git_commit_sha, hash_json
    if reference_set is None and not dataconfigs:
        raise ValueError("save_dnalignair_bundle needs either dataconfigs or reference_set")
    d = Path(bundle_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    config_dict = model.config.to_dict()
    (d / "config.json").write_text(json.dumps(config_dict, indent=2, sort_keys=True))
    if reference_set is not None:
        ref_payload = {**_embed_reference(reference_set), "locus": locus}
    else:
        ref_payload = {"dataconfigs": list(dataconfigs), "locus": locus}
    (d / "reference.json").write_text(json.dumps(ref_payload, indent=2, sort_keys=True))
    if calibration is not None:
        (d / "calibration.json").write_text(json.dumps(calibration, indent=2, sort_keys=True))
    meta = {
        "format_version": DNALIGNAIR_BUNDLE_VERSION,
        "notes": notes,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "alignair_version": alignair_version(),
        "git_commit": git_commit_sha(),
        "versions": package_versions(),
        "reference_hash": hash_json(ref_payload),
        "config_hash": hash_json(config_dict),
        "calibration_hash": hash_json(calibration),
        "training": dict(training_meta or {}),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (d / "VERSION").write_text(str(DNALIGNAIR_BUNDLE_VERSION))
    (d / "fingerprint.txt").write_text(compute_fingerprint(d))   # written last; excluded from itself
    return str(d)


def load_dnalignair_bundle(bundle_dir, *, build: bool = True, device: str = "cpu") -> dict:
    """Load a bundle. Returns {config, dataconfigs, locus, calibration, meta[, model]}.
    Verifies the fingerprint (raises on tamper/corruption)."""
    d = Path(bundle_dir)
    missing = [n for n in _REQUIRED if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(f"DNAlignAIR bundle missing required files: {missing}")
    if compute_fingerprint(d) != (d / "fingerprint.txt").read_text().strip():
        raise ValueError(f"bundle fingerprint mismatch — {d} was modified or is corrupt")

    config = DNAlignAIRConfig(**json.loads((d / "config.json").read_text()))
    ref = json.loads((d / "reference.json").read_text())
    calibration = (json.loads((d / "calibration.json").read_text())
                   if (d / "calibration.json").exists() else None)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    # embedded (custom) reference -> rebuild a ReferenceSet directly; else keep dataconfig names.
    reference_set = None
    if "genotype" in ref:
        from ..reference.reference_set import ReferenceSet
        reference_set = ReferenceSet.from_genotype(ref["genotype"], anchors=ref.get("anchors"))
    out = {"config": config, "dataconfigs": ref.get("dataconfigs"),
           "reference_set": reference_set, "locus": ref.get("locus", "IGH"),
           "calibration": calibration, "meta": meta}
    if build:
        from ..core.dnalignair import DNAlignAIR
        model = DNAlignAIR(config)
        model.load_state_dict(torch.load(d / "model.pt", map_location=device, weights_only=True))
        out["model"] = model.to(device).eval()
    return out


def is_bundle(path: str) -> bool:
    """True if `path` is a directory that looks like a DNAlignAIR bundle."""
    p = Path(path)
    return p.is_dir() and (p / "config.json").exists() and (p / "model.pt").exists()
