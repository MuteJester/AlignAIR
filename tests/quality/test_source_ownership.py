"""Keep retired namespaces out of live package imports after capability migration, and the development packages
(the assay, the lab) out of the product."""
import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2] / 'src' / 'alignair'


def _imports(root):
    """``(path relative to root, line, absolute dotted name)`` of every import of every module under ``root``."""
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
                yield path.relative_to(root), node.lineno, name


def test_live_sources_use_canonical_capability_namespaces():
    assert not (ROOT / 'aa').exists()
    assert not (ROOT / 'nn').exists()
    assert not (ROOT / 'core' / 'losses.py').exists()
    retired = ('alignair.aa', 'alignair.nn', 'alignair.core.losses')
    failures = [f'{path}:{line}: {name}' for path, line, name in _imports(ROOT)
                if any(name == old or name.startswith(old + '.') for old in retired)]
    assert not failures, '\n'.join(failures)


def test_the_product_imports_neither_the_assay_nor_the_lab():
    development = ('alignair_assay', 'alignair_lab')
    failures = [f'{path}:{line}: {name}' for path, line, name in _imports(ROOT)
                if name.split('.')[0] in development]
    assert not failures, '\n'.join(failures)
