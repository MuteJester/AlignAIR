"""Versioned, self-describing bundle for a trained DNAlignAIR model.

A bundle is a directory that packages everything needed to deploy one model:
    model.pt          state_dict
    config.json       DNAlignAIRConfig
    reference.json    {"dataconfigs": [...names...], "locus": "IGH"}  (default reference)
    calibration.json  per-gene equivalence-set calibration (optional)
    meta.json         {format_version, notes}
    VERSION
    fingerprint.txt   SHA-256 over the other files (tamper detection)

This is distinct from serialization/bundle.py, which targets the legacy ModelConfig /
SingleChain·MultiChain lineage. The module stays free of GenAIRR — it stores dataconfig
NAMES; the caller (CLI) reconstructs the ReferenceSet from them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import torch

from ..config.dnalignair_config import DNAlignAIRConfig
from .bundle import compute_fingerprint

DNALIGNAIR_BUNDLE_VERSION = 1
_REQUIRED = ("model.pt", "config.json", "reference.json", "VERSION", "fingerprint.txt")


def save_dnalignair_bundle(bundle_dir, *, model, dataconfigs: Iterable[str], locus: str = "IGH",
                           calibration: Optional[dict] = None, notes: Optional[str] = None) -> str:
    """Write a bundle for `model` (a DNAlignAIR with .config and .state_dict()).
    `dataconfigs` are GenAIRR DataConfig NAMES used to build the default reference."""
    d = Path(bundle_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    (d / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2, sort_keys=True))
    (d / "reference.json").write_text(
        json.dumps({"dataconfigs": list(dataconfigs), "locus": locus}, indent=2, sort_keys=True))
    if calibration is not None:
        (d / "calibration.json").write_text(json.dumps(calibration, indent=2, sort_keys=True))
    (d / "meta.json").write_text(
        json.dumps({"format_version": DNALIGNAIR_BUNDLE_VERSION, "notes": notes}, indent=2, sort_keys=True))
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
    out = {"config": config, "dataconfigs": ref["dataconfigs"], "locus": ref.get("locus", "IGH"),
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
