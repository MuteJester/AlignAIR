"""The shipped package passes ruff under pyproject.toml's ``[tool.ruff]`` settings, so a new finding fails the suite.
The ``test`` extra pins ruff's version, the one this test runs: another minor version may enable other default rules.
Outside CI a missing or other ruff skips; in CI (``CI`` set) it fails, so the gate cannot turn itself off."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2]


def _pin():
    """The ``test`` extra's ruff requirement's version specifier."""
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["test"]
    return next(requirement.specifier for requirement in map(Requirement, extra) if requirement.name == "ruff")


def _ruff() -> list[str] | None:
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    found = shutil.which("ruff")
    return [found] if found else None


def _unavailable(reason: str):
    if os.environ.get("CI"):
        pytest.fail(f"the lint gate needs the pinned ruff in CI: {reason}")
    pytest.skip(reason)


def test_the_package_passes_ruff():
    pin, ruff = _pin(), _ruff()
    if ruff is None:
        _unavailable(f"ruff is not installed (pip install 'AlignAIR[test]', ruff{pin})")
    version = subprocess.run([*ruff, "--version"], capture_output=True, text=True, check=True).stdout.split()[-1]
    if version not in pin:
        _unavailable(f"ruff {version} is not the pinned ruff{pin}")
    result = subprocess.run([*ruff, "check", "--no-cache", "src/alignair"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
