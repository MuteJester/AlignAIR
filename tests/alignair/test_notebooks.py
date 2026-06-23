"""Execute the example notebooks so they can't go stale (#104).

Skipped unless the notebook tooling is installed (it is in the `dev` extra, not `cli`),
so CI's lean test job skips this while a full `pip install .[dev]` run exercises it.
"""
import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")
pytest.importorskip("torch")
pytest.importorskip("GenAIRR")

from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes(nb_path, tmp_path, monkeypatch):
    # Run in a temp dir (notebooks write demo_out/, cr_model/, ... to cwd) with an `examples`
    # symlink so the notebooks' repo-root-relative paths resolve — keeps the tree clean.
    (tmp_path / "examples").symlink_to(ROOT / "examples")
    # The kernel spawns with cwd=tmp_path, so a relative PYTHONPATH=src would break the
    # `python -m alignair.cli` cells (a pip install would not need this).
    src = ROOT / "src"
    if src.is_dir():
        existing = os.environ.get("PYTHONPATH", "")
        monkeypatch.setenv("PYTHONPATH", str(src) + (os.pathsep + existing if existing else ""))
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}})
    client.execute()                                                       # raises on any cell error
