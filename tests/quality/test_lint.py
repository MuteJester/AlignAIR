"""The shipped package passes ruff under pyproject.toml's ``[tool.ruff]`` settings, so a new finding fails the suite.
The ``test`` extra pins ruff's minor version: another minor version may enable other default rules, and is skipped."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2]
PINNED = "0.17."            # the `test` extra's ruff pin (pyproject.toml)


def _ruff() -> list[str] | None:
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    found = shutil.which("ruff")
    return [found] if found else None


def test_the_package_passes_ruff():
    ruff = _ruff()
    if ruff is None:
        pytest.skip("ruff is not installed (pip install 'AlignAIR[test]')")
    version = subprocess.run([*ruff, "--version"], capture_output=True, text=True, check=True).stdout.split()[-1]
    if not version.startswith(PINNED):
        pytest.skip(f"ruff {version} is not the pinned {PINNED}x")
    result = subprocess.run([*ruff, "check", "--no-cache", "src/alignair"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
