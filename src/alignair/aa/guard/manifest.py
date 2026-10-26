"""Frozen-cohort manifest self-seal hashing — one documented canonicalization rule, used by both the
writer (Stage-1 gate) and any reader/verifier so the hash is reproducible.

CANONICALIZATION RULE (non-recursive):

    manifest_sha256 = sha256( json.dumps(M', sort_keys=True, separators=(",", ":")).encode() ).hexdigest()

where M' is the manifest dict with the ``manifest_sha256`` key REMOVED. The field is omitted from its own
input — the hash never recurses into itself. Keys are sorted and separators are compact (no whitespace) so
the byte stream is independent of indentation or key order in the stored file. A reader recomputes the hash
with this exact rule and compares to the stored value.
"""
from __future__ import annotations

import hashlib
import json

MANIFEST_HASH_FIELD = "manifest_sha256"


def canonical_bytes(manifest: dict) -> bytes:
    """Canonical byte stream for hashing: sorted-key, compact-separator JSON with the hash field removed."""
    body = {k: v for k, v in manifest.items() if k != MANIFEST_HASH_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def compute_manifest_hash(manifest: dict) -> str:
    """sha256 of the canonical bytes (manifest_sha256 omitted from its own input)."""
    return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def seal_manifest(manifest: dict) -> dict:
    """Return the manifest with its manifest_sha256 field set to the computed hash."""
    manifest = dict(manifest)
    manifest[MANIFEST_HASH_FIELD] = compute_manifest_hash(manifest)
    return manifest


def verify_manifest(manifest: dict) -> bool:
    """True iff the stored manifest_sha256 equals the recomputed hash under the documented rule."""
    return manifest.get(MANIFEST_HASH_FIELD) == compute_manifest_hash(manifest)
