"""The model gate rejects unavailable or altered evidence before loading weights."""
import hashlib
import json

import pytest

from tests.support.model_artifacts import provision, read_manifest, verified_path

pytestmark = pytest.mark.component


@pytest.fixture
def artifact(tmp_path):
    content = b"pinned model evidence"
    entry = {"file": "model/1.alignair", "id": "model", "version": "1",
             "artifact_sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    path = tmp_path / entry["file"]
    path.parent.mkdir()
    path.write_bytes(content)
    return entry, path


def test_verified_fixture_accepts_exact_bytes(tmp_path, artifact):
    entry, path = artifact
    assert verified_path(tmp_path, entry) == path


@pytest.mark.parametrize("corruption", ["missing", "truncated", "same-size"])
def test_gate_rejects_missing_or_corrupt_fixture(tmp_path, artifact, corruption):
    entry, path = artifact
    if corruption == "missing":
        path.unlink()
    elif corruption == "truncated":
        path.write_bytes(b"short")
    else:
        path.write_bytes(b"x" * entry["size"])
    with pytest.raises(ValueError, match="Required model fixture"):
        verified_path(tmp_path, entry)


@pytest.mark.parametrize("file", ["../escape.alignair", "/escape.alignair",
                                  "..\\escape.alignair", "C:/escape.alignair", ""])
def test_manifest_rejects_paths_outside_the_fixture_cache(tmp_path, artifact, file):
    entry, _ = artifact
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "alignair.test-models.v1", "source": "file://fixture",
                               "models": {"model": {**entry, "file": file}}}))
    with pytest.raises(ValueError, match="invalid fixture path"):
        read_manifest(path)


def test_provisioning_uses_the_runtime_verified_downloader(tmp_path, artifact):
    entry, original = artifact
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps({"schema": "alignair.test-models.v1", "source": tmp_path.as_uri(),
                                  "models": {"model": entry}}))
    cache = tmp_path / "cache"
    provision(cache, source)
    assert verified_path(cache, entry).read_bytes() == original.read_bytes()
    # A second pass must succeed from the verified cache with the source removed.
    original.unlink()
    provision(cache, source)
