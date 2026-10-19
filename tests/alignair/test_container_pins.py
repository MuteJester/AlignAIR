"""Every pinned container reference in the docs and workflow wrappers must match pyproject's
version, so a release can't leave them pulling the previous image."""
try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
PIN = re.compile(r"(?:alignair/alignair|ghcr\.io/mutejester/alignair):(\d[\w.+-]*)")

PINNED_FILES = [
    "README.md",
    "examples/README.md",
    "workflows/README.md",
    "workflows/galaxy/alignair_predict.xml",
    "workflows/nextflow/main.nf",
    "workflows/snakemake/Snakefile",
    "workflows/snakemake/config.yaml",
    "website/src/pages/docs/pages-getstarted.tsx",
    "website/src/pages/docs/pages-guides2.tsx",
    "website/src/pages/docs/pages-reference.tsx",
]


@pytest.mark.parametrize("rel", PINNED_FILES)
def test_container_pins_match_pyproject_version(rel):
    pins = PIN.findall((ROOT / rel).read_text())
    stale = sorted({p for p in pins if p != VERSION})
    assert not stale, f"{rel} pins {stale}, expected {VERSION}"


def test_wrappers_pin_a_version_not_latest():
    for rel in ("workflows/nextflow/main.nf", "workflows/snakemake/config.yaml",
                "workflows/galaxy/alignair_predict.xml"):
        text = (ROOT / rel).read_text()
        assert f"alignair/alignair:{VERSION}" in text, rel
        assert "thomask90/alignair" not in text, rel
