"""Shared paths for fixtures used across package and packaging tests."""
from pathlib import Path

import pytest

pytest_plugins = (
    "pytester",
    "tests.support.environment",
    "tests.support.policy",
    "tests.support.model_artifacts",
)


@pytest.fixture(scope="session")
def repo_root(pytestconfig) -> Path:
    return pytestconfig.rootpath


@pytest.fixture(scope="session")
def training_csv(repo_root) -> Path:
    """The retained IGH CSV also tests real allele-vocabulary cardinalities."""
    return repo_root / "tests" / "fixtures" / "training" / "sample_igh.csv"
