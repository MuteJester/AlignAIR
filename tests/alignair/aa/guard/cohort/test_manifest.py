import json
from pathlib import Path

import pytest

from alignair.aa.guard.manifest import (MANIFEST_HASH_FIELD, compute_manifest_hash,
                                        seal_manifest, verify_manifest)

pytestmark = pytest.mark.component

FROZEN = Path("Notes and Plans/docs/superpowers/artifacts/2026-07-26-aa-applicability/FROZEN_COHORT_MANIFEST.json")


def test_seal_and_verify_roundtrip():
    sealed = seal_manifest({"b": 2, "a": [1, {"z": 9}], "note": "x"})
    assert sealed[MANIFEST_HASH_FIELD] == compute_manifest_hash(sealed)
    assert verify_manifest(sealed)


def test_hash_is_non_recursive():
    # the hash field is omitted from its own input, so tampering it never changes the recomputed hash
    sealed = seal_manifest({"a": 1})
    tampered = dict(sealed, **{MANIFEST_HASH_FIELD: "deadbeef"})
    assert compute_manifest_hash(tampered) == compute_manifest_hash(sealed)
    assert not verify_manifest(tampered)


def test_hash_changes_when_content_changes():
    assert seal_manifest({"x": 1})[MANIFEST_HASH_FIELD] != seal_manifest({"x": 2})[MANIFEST_HASH_FIELD]


def test_canonicalization_is_indent_and_order_independent():
    # same content, different key order -> identical canonical hash
    assert compute_manifest_hash({"a": 1, "b": 2}) == compute_manifest_hash({"b": 2, "a": 1})


@pytest.mark.local_data
def test_frozen_manifest_file_reproduces_its_hash():
    if not FROZEN.exists():
        pytest.skip("cohort manifest not frozen in this checkout")
    m = json.loads(FROZEN.read_text())
    assert verify_manifest(m), "stored manifest_sha256 does not reproduce under the documented rule"
