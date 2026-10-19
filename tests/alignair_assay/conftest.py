"""Shared fixtures for the assay tests: the GenAIRR guard, the truth reference and the committed bench cases."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignair_assay.core.reference import Reference

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE_NAME = "HUMAN_IGH_OGRDB"


@pytest.fixture(scope="session")
def genairr():
    """The GenAIRR package; a test that needs it, directly or through ``reference``, is skipped when it is missing."""
    return pytest.importorskip("GenAIRR")


@pytest.fixture(scope="session")
def reference(genairr):
    """The truth reference (``HUMAN_IGH_OGRDB``), built once per test session."""
    return Reference.from_dataconfig(REFERENCE_NAME)


@pytest.fixture(scope="session")
def load_fixture():
    """``load_fixture(name) -> dict``: the committed bench case ``fixtures/<name>``."""
    def load(name: str) -> dict:
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    return load
