"""Keep retired namespaces out of live package imports after capability migration."""
import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.component


def test_live_sources_use_canonical_capability_namespaces():
    root = Path(__file__).resolve().parents[2] / 'src' / 'alignair'
    assert not (root / 'aa').exists()
    assert not (root / 'nn').exists()
    assert not (root / 'core' / 'losses.py').exists()
    retired = ('alignair.aa', 'alignair.nn', 'alignair.core.losses')
    failures = []
    for path in root.rglob('*.py'):
        package = 'alignair.' + '.'.join(path.relative_to(root).with_suffix('').parts[:-1])
        package = package.rstrip('.')
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                         if node.level else node.module or '']
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if any(name == old or name.startswith(old + '.') for old in retired):
                    failures.append(f'{path.relative_to(root)}:{node.lineno}: {name}')
    assert not failures, '\n'.join(failures)
