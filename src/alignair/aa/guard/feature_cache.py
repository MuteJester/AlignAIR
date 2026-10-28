"""Identity-bound HMM feature preparation, separate from training targets and tensor assembly."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

from alignair.errors import ArtifactIntegrityError, ConfigurationError, DataContractError, InputError, OperationIOError, error_context
from alignair.io.atomic import atomic_write_json
from .features import FEATURE_NAMES, hmm_features

CACHE_SCHEMA_VERSION = 1


def _mask_groups(value):
    try:
        if isinstance(value, (str, bytes)):
            raise TypeError
        groups = frozenset(value)
        if any(not isinstance(group, str) or not group for group in groups):
            raise TypeError
    except TypeError as exc:
        raise ConfigurationError("Masked profile groups must be an iterable of nonempty names.") from exc
    return groups


def sequence_key(sequence):
    """Hash the exact supplied sequence; do not change case, strip or truncate it."""
    if not isinstance(sequence, str) or not sequence:
        raise InputError("Feature preparation requires nonempty sequence text.", context={"field": "aa"})
    return hashlib.sha256(sequence.encode()).hexdigest()


def _vector(value):
    # Preserve native int/float encodings without accepting numeric strings or booleans.
    if not isinstance(value, (list, tuple)) or len(value) != len(FEATURE_NAMES):
        raise DataContractError("Guard feature vector has the wrong width.")
    normalized = []
    for item in value:
        try:
            valid = isinstance(item, Real) and not isinstance(item, bool) and math.isfinite(item)
        except OverflowError:
            valid = False
        if not valid:
            raise DataContractError("Guard feature vectors require finite numeric values.")
        normalized.append(int(item) if isinstance(item, Integral) else float(item))
    return tuple(normalized)


@dataclass(frozen=True)
class FeatureCache(Mapping):
    """Read-only sequence-hash mapping retaining bank, feature order and profile-mask identity.

    Indexing returns a list copy. Use dict(cache) for the JSON-compatible vector mapping.
    Identity metadata establishes compatibility, not authenticity of supplied feature values.
    """

    vectors: Mapping = field(repr=False)
    bank_checksum: str
    mask_groups: frozenset = frozenset()
    feature_names: tuple = field(default_factory=lambda: tuple(FEATURE_NAMES))

    def __post_init__(self):
        if not isinstance(self.bank_checksum, str) or not self.bank_checksum:
            raise ConfigurationError("Feature cache requires a bank checksum.")
        if not isinstance(self.feature_names, (list, tuple)) or tuple(self.feature_names) != tuple(FEATURE_NAMES):
            raise ArtifactIntegrityError("Guard feature ordering differs from the current extractor.")
        if not isinstance(self.vectors, Mapping):
            raise DataContractError("Guard feature cache vectors must be a mapping.")
        vectors = {}
        for key, value in self.vectors.items():
            with error_context(sequence_hash=key):
                if not isinstance(key, str) or len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
                    raise DataContractError("Guard feature cache keys must be sequence SHA-256 digests.")
                vectors[key] = _vector(value)
        object.__setattr__(self, "vectors", MappingProxyType(vectors))
        object.__setattr__(self, "mask_groups", _mask_groups(self.mask_groups))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    def __getitem__(self, key):
        return list(self.vectors[key])

    def __iter__(self):
        return iter(self.vectors)

    def __len__(self):
        return len(self.vectors)


def bind_feature_cache(cache, *, bank_checksum, mask_groups=frozenset()):
    """Check a bound cache, or validate and explicitly bind a caller-owned vector mapping."""
    with error_context(stage="guard_feature_cache"):
        if not isinstance(bank_checksum, str) or not bank_checksum:
            raise ConfigurationError("Feature cache requires an expected bank checksum.")
        groups = _mask_groups(mask_groups)
        if isinstance(cache, FeatureCache):
            expected = {"bank_checksum": bank_checksum, "mask_groups": groups, "feature_names": tuple(FEATURE_NAMES)}
            mismatch = [name for name, value in expected.items() if getattr(cache, name) != value]
            if mismatch:
                raise ArtifactIntegrityError("Guard feature cache identity differs from the requested bank, ordering or masks.",
                                             context={"fields": mismatch})
            return cache
        return FeatureCache(cache, bank_checksum=bank_checksum, mask_groups=groups)


def feature_vector(cache: FeatureCache, sequence):
    key = sequence_key(sequence)
    try:
        return cache.vectors[key]
    except KeyError as exc:
        raise DataContractError("Guard features were not prepared for this sequence.",
                                context={"stage": "guard_feature_cache", "sequence_hash": key}) from exc


def precompute_features(records, bank, *, bank_checksum, cache_path, mask_groups=frozenset()):
    """Reuse matching validated vectors, extend missing sequences, or rebuild stale identities.

    Legacy unbound and mismatched-identity caches are rebuilt. Malformed matching caches are
    refused without replacing their files. Only a complete successful preparation is published.
    """
    path = Path(cache_path)
    with error_context(stage="guard_feature_cache", path=str(path)):
        groups = _mask_groups(mask_groups)
        if not isinstance(bank_checksum, str) or not bank_checksum or not callable(getattr(bank, "checksum", None)):
            raise ConfigurationError("Feature preparation requires a profile bank and checksum.")
        if bank.checksum() != bank_checksum:
            raise ArtifactIntegrityError("Profile bank differs from the requested feature cache identity.")
        sequences = {}
        for index, record in enumerate(records):
            with error_context(record_index=index):
                if not isinstance(record, Mapping):
                    raise InputError("Feature records must be mappings.")
                sequence = record.get("aa")
                sequences.setdefault(sequence_key(sequence), sequence)
        identity = {"schema_version": CACHE_SCHEMA_VERSION, "bank_checksum": bank_checksum,
                    "feature_names": list(FEATURE_NAMES), "mask_groups": sorted(groups)}
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            stored = {}
        except ValueError as exc:
            raise DataContractError("Invalid guard feature cache JSON.") from exc
        except OSError as exc:
            raise OperationIOError("Cannot read guard feature cache.") from exc
        if not isinstance(stored, dict):
            raise DataContractError("Guard feature cache must be a mapping.")
        matched = type(stored.get("schema_version")) is int and all(stored.get(key) == value for key, value in identity.items())
        cache = bind_feature_cache(stored.get("vectors") if matched else {},
                                   bank_checksum=bank_checksum, mask_groups=groups)
        missing = [(key, sequence) for key, sequence in sequences.items() if key not in cache]
        if matched and not missing:
            return cache
        vectors = dict(cache.vectors)
        for key, sequence in missing:
            with error_context(sequence_hash=key):
                vectors[key] = _vector(hmm_features(bank, sequence, mask_groups=groups).to_vector().tolist())
        prepared = FeatureCache(vectors, bank_checksum=bank_checksum, mask_groups=groups)
        atomic_write_json(path, {**identity, "vectors": dict(prepared.vectors)}, sort_keys=False, indent=None,
                          stage="guard_feature_cache")
        return prepared
