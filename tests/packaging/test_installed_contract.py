"""The wheel job's contract script (``installed_contract.py``) runs here on the source tree, so a public name it still
calls after the name is removed fails the suite, not only the wheel and release jobs. Its two "this is an installed
copy" checks are pointed at the source tree, and the script imports the alignair this test imported (its package root
first on ``PYTHONPATH``: in CI an installed alignair is importable too); everything after them runs as in CI."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "packaging" / "installed_contract.py"


def test_the_installed_contract_script_runs_on_the_source_tree(tmp_path):
    import alignair
    package_root = str(Path(alignair.__file__).resolve().parents[1])
    code = f"""
import importlib.util, runpy, sys, sysconfig
real_get_path, real_find_spec = sysconfig.get_path, importlib.util.find_spec
sysconfig.get_path = lambda name, *a, **k: {package_root!r} if name in ("purelib", "platlib") else real_get_path(name, *a, **k)
importlib.util.find_spec = lambda name, *a, **k: None if name == "alignair_lab" else real_find_spec(name, *a, **k)
sys.argv = ["installed_contract.py", "--forbid-source", {str(tmp_path)!r}]
runpy.run_path({str(SCRIPT)!r}, run_name="__main__")
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [package_root, os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "Installed distribution contracts passed" in result.stdout
