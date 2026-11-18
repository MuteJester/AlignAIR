"""The public package works without pandas or airr installed."""
import pytest
import subprocess
import sys
import textwrap

pytestmark = pytest.mark.component


def test_package_and_cli_without_pandas(repo_root):
    code = textwrap.dedent('''
        import importlib.abc
        import importlib.machinery
        import importlib.util
        import sys

        sys.path.insert(0, "src")

        class NoPandasOrAirr(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"pandas", "airr"}:
                    return None
                return importlib.machinery.PathFinder.find_spec(fullname, path, target)

        # Replace the path finder so find_spec also reports an absent package, as it
        # would on a clean install (PyTorch probes optional packages this way).
        sys.meta_path = [NoPandasOrAirr() if finder is importlib.machinery.PathFinder else finder
                         for finder in sys.meta_path]
        assert importlib.util.find_spec("pandas") is None
        assert importlib.util.find_spec("airr") is None
        import alignair
        from alignair.cli.main import build_parser
        from alignair.genotype.airr import validate

        build_parser()
        import torch
        model = torch.nn.Linear(2, 1)
        torch.optim.AdamW(model.parameters(), lr=1e-4)
        assert validate({})
        assert "pandas" not in sys.modules
        assert "airr" not in sys.modules
    ''')
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
