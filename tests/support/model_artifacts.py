"""Provision public model fixtures before pytest; tests consume verified files offline."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import pytest

MANIFEST = Path(__file__).resolve().parents[1] / "fixtures" / "models" / "validation.json"


def read_manifest(path=MANIFEST):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "alignair.test-models.v1" or not data.get("source"):
        raise ValueError("Unsupported model-validation manifest")
    if not isinstance(data.get("models"), dict) or not data["models"]:
        raise ValueError("Model-validation manifest has no models")
    for name, entry in data["models"].items():
        relative = PurePosixPath(entry["file"])
        if (relative.is_absolute() or ".." in relative.parts or "\\" in entry["file"]
                or ":" in entry["file"] or relative.suffix != ".alignair"):
            raise ValueError(f"{name}: invalid fixture path")
        if not re.fullmatch(r"[a-f0-9]{64}", entry["artifact_sha256"]):
            raise ValueError(f"{name}: invalid artifact checksum")
        if type(entry["size"]) is not int or entry["size"] <= 0:
            raise ValueError(f"{name}: invalid artifact size")
    return data


def verified_path(cache, entry):
    root = Path(cache).resolve()
    path = (root / entry["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Model fixture escapes its cache")
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise ValueError(f"Required model fixture missing or wrong size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != entry["artifact_sha256"]:
        raise ValueError(f"Required model fixture checksum mismatch: {path}")
    return path


def provision(cache, manifest=MANIFEST):
    from alignair.registry.cache import download_verified
    data = read_manifest(manifest)
    for entry in data["models"].values():
        dest = Path(cache) / entry["file"]
        if not dest.resolve().is_relative_to(Path(cache).resolve()):
            raise ValueError("Model fixture escapes its cache")
        download_verified(data["source"], entry["file"], dest,
                          expected_sha256=entry["artifact_sha256"])
        verified_path(cache, entry)
        print(f"Verified {entry['id']}@{entry['version']}")


@pytest.fixture(scope="session")
def validation_models(pytestconfig):
    cache = pytestconfig.getoption("--model-cache")
    if not cache:
        message = "Model validation requires --model-cache with pre-fetched pinned artifacts"
        if pytestconfig.getoption("--enforce-claims"):
            pytest.fail(message)
        pytest.skip(message)
    data = read_manifest()
    try:
        return {key: verified_path(cache, entry) for key, entry in data["models"].items()}
    except ValueError as exc:
        pytest.fail(str(exc))


@pytest.fixture
def igh_model(validation_models):
    return str(validation_models["igh"])


@pytest.fixture
def igkl_model(validation_models):
    return str(validation_models["igkl"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    provision(args.cache, args.manifest)


if __name__ == "__main__":
    main()
