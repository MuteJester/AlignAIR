"""Validation utilities for model bundle integrity and compatibility."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Dict, Any
import json
import pickle
import hashlib
import logging

from .model_bundle import FORMAT_VERSION, ModelBundleConfig

logger = logging.getLogger(__name__)

REQUIRED_FILES = [
    "config.json",
    "dataconfig.pkl",
    "fingerprint.txt",
    "VERSION",
]


def ensure_required_files(bundle_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (bundle_dir / f).exists()]
    # Additionally require SavedModel directory
    if not (bundle_dir / 'saved_model').exists():
        missing.append('saved_model')
    if missing:
        raise FileNotFoundError(f"Missing required bundle file(s): {missing}. "
                                f"Ensure the bundle directory is complete and not corrupted.")


def load_config(bundle_dir: Path) -> ModelBundleConfig:
    cfg_path = bundle_dir / "config.json"
    with cfg_path.open("r") as f:
        raw = json.load(f)
    if "format_version" not in raw:
        raise ValueError("config.json missing 'format_version'. Regenerate bundle with save_pretrained().")
    if raw["format_version"] > FORMAT_VERSION:
        raise RuntimeError(
            f"Bundle format_version={raw['format_version']} > supported={FORMAT_VERSION}. "
            f"Upgrade AlignAIRR to a newer version to load this bundle.")
    try:
        return ModelBundleConfig.from_dict(raw)
    except TypeError as e:
        raise ValueError(f"config.json schema mismatch: {e}") from e


def verify_version_file(bundle_dir: Path, config: ModelBundleConfig) -> None:
    version_file = bundle_dir / "VERSION"
    text = version_file.read_text().strip()
    expected = f"FORMAT_VERSION={config.format_version}"
    if text != expected:
        raise ValueError(f"VERSION file mismatch: found '{text}' expected '{expected}'.")


def compute_fingerprint(bundle_dir: Path) -> str:
    sha = hashlib.sha256()
    # Include structural files
    for name in ["config.json", "dataconfig.pkl"]:
        with (bundle_dir / name).open("rb") as f:
            sha.update(f.read())
    # Include SavedModel assets deterministically (sort paths)
    sm_dir = bundle_dir / 'saved_model'
    if sm_dir.exists():
        for p in sorted(sm_dir.rglob('*')):
            if p.is_file():
                sha.update(p.relative_to(bundle_dir).as_posix().encode('utf-8'))
                with p.open('rb') as f:
                    sha.update(f.read())
    return sha.hexdigest()


def verify_fingerprint(bundle_dir: Path) -> None:
    expected = (bundle_dir / "fingerprint.txt").read_text().strip()
    actual = compute_fingerprint(bundle_dir)
    if expected != actual:
        raise ValueError(
            "Fingerprint mismatch: expected {} got {}. The bundle may be corrupted or partially modified. "
            "Re-download or regenerate using save_pretrained().".format(expected, actual)
        )


def validate_dataconfig_compat(config: ModelBundleConfig, dataconfig_obj: Any) -> None:
    """Validate allele count & chain-type structural compatibility.

    Parameters
    ----------
    config : ModelBundleConfig
        Saved structural configuration.
    dataconfig_obj : Any
        Either a single DataConfig or a MultiDataConfigContainer.
    """
    # Lazy imports to keep base import path light
    from GenAIRR.dataconfig import DataConfig  # type: ignore
    from AlignAIR.Data import MultiDataConfigContainer  # type: ignore

    if config.model_type == 'single_chain':
        if not isinstance(dataconfig_obj, (DataConfig,)):  # could be container wrapper internally
            # Some code wraps single inside MultiDataConfigContainer; allow attribute access fallback
            pass
        # Expect the dataconfig to expose these counts
        v = getattr(dataconfig_obj, 'number_of_v_alleles', None)
        j = getattr(dataconfig_obj, 'number_of_j_alleles', None)
        d = getattr(dataconfig_obj, 'number_of_d_alleles', None)
        mismatches = []
        if v is not None and v != config.v_allele_count:
            mismatches.append(f"v_allele_count config={config.v_allele_count} dataconfig={v}")
        if j is not None and j != config.j_allele_count:
            mismatches.append(f"j_allele_count config={config.j_allele_count} dataconfig={j}")
        if config.has_d_gene and d is not None and d != config.d_allele_count:
            mismatches.append(f"d_allele_count config={config.d_allele_count} dataconfig={d}")
        if mismatches:
            raise ValueError("Dataconfig mismatch: " + "; ".join(mismatches))
    else:  # multi_chain
        if not hasattr(dataconfig_obj, 'chain_types'):
            raise ValueError("Expected MultiDataConfigContainer with chain_types() method for multi_chain bundle.")
        # Normalize both sides to string values to avoid issues with Enum comparisons
        chain_types = [getattr(ct, 'value', str(ct)) for ct in dataconfig_obj.chain_types()]
        config_chain_types = [str(ct) for ct in (config.chain_types or [])]
        if config.chain_types and sorted(chain_types) != sorted(config_chain_types):
            raise ValueError(f"Chain types differ: config={config_chain_types} actual={chain_types}")
        # Aggregate counts
        v = getattr(dataconfig_obj, 'number_of_v_alleles', None)
        j = getattr(dataconfig_obj, 'number_of_j_alleles', None)
        d = getattr(dataconfig_obj, 'number_of_d_alleles', None)
        mismatches = []
        if v is not None and v != config.v_allele_count:
            mismatches.append(f"v_allele_count config={config.v_allele_count} dataconfigs={v}")
        if j is not None and j != config.j_allele_count:
            mismatches.append(f"j_allele_count config={config.j_allele_count} dataconfigs={j}")
        # d is optional
        if config.has_d_gene and d is not None and d != config.d_allele_count:
            mismatches.append(f"d_allele_count config={config.d_allele_count} dataconfigs={d}")
        if mismatches:
            raise ValueError("Dataconfig mismatch: " + "; ".join(mismatches))


__all__ = [
    "ensure_required_files",
    "load_config",
    "verify_version_file",
    "verify_fingerprint",
    "validate_dataconfig_compat",
    "compute_fingerprint",
]
