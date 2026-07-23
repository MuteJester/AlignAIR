"""AA `.alignair` bundle writer: safetensors weights + canonical-JSON sections (config, tokenizer,
class_maps, asc_imgt, calibration) over `model_file.container`, with a card + integrity-manifest
header. Zero pickle. The card deliberately carries NO `host`/`user` fields (those would leak
build-machine / maintainer identity; per spec §3 they stay in a private, uncommitted build log)."""
from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime, timezone

import torch

from ...model_file import container, serialize

BUNDLE_SCHEMA_VERSION = "alignair.aa.bundle.v1"

# §3: alphabet + special-token id map, frozen so decoding is reproducible from the bundle alone.
_TOKENIZER_SECTION = {
    "alphabet": "ACDEFGHIKLMNPQRSTVWY",
    "special": {"PAD": 0, "X": 21, "*": 22},
    "vocab_size": 23,
}

# codec per section: raw weights are already compact (safetensors); everything else is JSON text.
_CODECS = {"weights": "none", "config": "zlib", "tokenizer": "zlib", "class_maps": "zlib",
           "asc_imgt": "zlib", "calibration": "zlib"}
_FORMATS = {"weights": "safetensors", "config": "json", "tokenizer": "json", "class_maps": "json",
            "asc_imgt": "json", "calibration": "json"}


def canonical_json(obj) -> bytes:
    """Deterministic JSON encoding: sorted keys, compact separators, UTF-8. Byte-identical across
    repeated builds of the same logical content, so it is safe to SHA-256 for the integrity manifest."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _alignair_version() -> str:
    try:
        from importlib.metadata import version
        return version("AlignAIR")
    except Exception:
        return "0+unknown"


def _build_commit() -> str | None:
    """Public git commit, best-effort (never fails the build if git/history is unavailable)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def complete_calibration(calibration: dict) -> dict:
    """Return a COPY of `calibration` with the conformal method parameters guaranteed present under a
    top-level `method` block: `{"name", "alpha", "lambda", "k_reg"}`. The frozen calibration JSONs
    (e.g. `calibration_cont_step318000.json`) predate this block entirely -- their calibrator
    (`aa/conformal.py`) ran with the `FrozenConformal` defaults (APS, i.e. `lambda=0.0`, `k_reg=1`) but
    never wrote those defaults back into the JSON, and `alpha` lives only at top level. This backfills
    exactly those gaps so Task-3's §4 binding (which requires an embedded `method/alpha/lambda/k_reg`)
    holds for artifacts built from those frozen snapshots, WITHOUT ever overwriting a value the source
    calibration already set explicitly (a future calibrator that writes its own `method` block wins)."""
    out = dict(calibration)
    method = dict(out.get("method") or {})
    method.setdefault("name", "APS")
    method.setdefault("alpha", calibration.get("alpha"))
    method.setdefault("lambda", 0.0)
    method.setdefault("k_reg", 1)
    out["method"] = method
    return out


def _calibration_provenance(calibration: dict) -> dict:
    """Pull method/alpha/lambda/k_reg/n_cal/n_test/supported_distribution from wherever they live in
    the calibration JSON. `method`/`lambda`/`k_reg`/(the fill for) `alpha` are resolved via
    `complete_calibration` -- so, per its contract, they are non-null defaults (APS/0.0/1) unless the
    source calibration set them explicitly. The frozen `calibration_cont_step318000.json` shape has
    `alpha`/`n_cal`/`n_test` at the top level and `supported_distribution` under `provenance`; a `meta`
    block (if a future calibration JSON carries one) takes priority for those three."""
    meta = calibration.get("meta") if isinstance(calibration.get("meta"), dict) else {}
    prov = calibration.get("provenance") if isinstance(calibration.get("provenance"), dict) else {}
    method = complete_calibration(calibration)["method"]
    return {
        "method": method["name"],
        "alpha": method["alpha"],
        "lambda": method["lambda"],
        "k_reg": method["k_reg"],
        "n_cal": _first(meta.get("n_cal"), calibration.get("n_cal"), prov.get("n_cal")),
        "n_test": _first(meta.get("n_test"), calibration.get("n_test"), prov.get("n_test")),
        "supported_distribution": _first(meta.get("supported_distribution"),
                                          calibration.get("supported_distribution"),
                                          prov.get("supported_distribution")),
    }


def save_aa_bundle(path, *, state_dict, cfg, class_maps, d_allele_order, asc_multi, calibration,
                    card_extra=None) -> dict:
    """Write a zero-pickle AA `.alignair` bundle to ``path``. Returns the written manifest (the header,
    freshly re-read off disk so it reflects exactly what was persisted).

    ``class_maps`` = ``{"V": EquivalenceClassMap, "J": EquivalenceClassMap}``; ``d_allele_order`` = the
    ordered list of D allele names (index == the model's D allele-head index); ``card_extra`` supplies
    ``model_id``, ``model_version``, ``calibration_scope`` (required — the §8 guarantee-scope string),
    ``source_model_sha256`` (required — the SHA-256 of the source ``.pt`` checkpoint file the weights
    were loaded from; this is the binding anchor a calibration JSON's ``provenance.model_sha256`` is
    checked against, since that hash is over raw ``.pt`` bytes and can never equal ``weights_sha256``
    (a hash over the re-serialized safetensors bytes) — two different hash domains over the same
    weights), plus optional ``created``/``license``/``citation``. Only these whitelisted fields are
    copied into the header, so arbitrary extra keys (e.g. accidental host/user) never leak into the
    card. The embedded ``calibration`` section is ``complete_calibration(calibration)``, not the raw
    input, so it always carries a ``method`` block (``name``/``alpha``/``lambda``/``k_reg``).
    """
    card_extra = card_extra or {}
    d_allele_order = list(d_allele_order)
    v_map, j_map = class_maps["V"], class_maps["J"]

    weights_bytes = serialize.state_dict_to_bytes(state_dict)
    config_bytes = canonical_json(dataclasses.asdict(cfg))
    tokenizer_bytes = canonical_json(_TOKENIZER_SECTION)
    class_maps_bytes = canonical_json(
        {"V": v_map.to_json(), "J": j_map.to_json(), "D_alleles": d_allele_order})
    asc_bytes = canonical_json(asc_multi)
    calibration_bytes = canonical_json(complete_calibration(calibration))

    sections = {
        "weights": (weights_bytes, _CODECS["weights"]),
        "config": (config_bytes, _CODECS["config"]),
        "tokenizer": (tokenizer_bytes, _CODECS["tokenizer"]),
        "class_maps": (class_maps_bytes, _CODECS["class_maps"]),
        "asc_imgt": (asc_bytes, _CODECS["asc_imgt"]),
        "calibration": (calibration_bytes, _CODECS["calibration"]),
    }

    header = {
        "artifact_type": "alignair-aa-model",
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "architecture_version": getattr(cfg, "architecture_version", None) or "aa-igh-v1",
        "model_id": card_extra["model_id"],
        "model_version": card_extra["model_version"],
        "created": card_extra.get("created") or datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"),
        "alignair_version": _alignair_version(),
        "torch_version": torch.__version__,
        "class_counts": {"v": v_map.n_classes(), "j": j_map.n_classes(), "d": len(d_allele_order)},
        "max_seq_length": cfg.max_seq_length,
        "calibration_scope": card_extra["calibration_scope"],
        "license": card_extra.get("license"),
        "citation": card_extra.get("citation"),
        "build_commit": _build_commit(),
        "reference_provenance": {
            "class_map_db_hash": {"V": v_map.db_hash, "J": j_map.db_hash},
            "class_map_sha256": container.sha256_hex(class_maps_bytes),
            "asc_table_sha256": container.sha256_hex(asc_bytes),
            "tokenizer_sha256": container.sha256_hex(tokenizer_bytes),
            "source_model_sha256": card_extra["source_model_sha256"],
        },
        "calibration_provenance": _calibration_provenance(calibration),
        "integrity": {
            "weights_sha256": container.sha256_hex(weights_bytes),
            "config_sha256": container.sha256_hex(config_bytes),
            "calibration_sha256": container.sha256_hex(calibration_bytes),
        },
        "_formats": _FORMATS,
    }
    assert "host" not in header and "user" not in header      # never leak build-machine identity

    container.write_container(str(path), header, sections)
    manifest = container.read_header(str(path))
    manifest.pop("_sections_base", None)
    return manifest
