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
