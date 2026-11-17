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


def _type_checking(node) -> bool:
    """Whether ``node`` is an ``if TYPE_CHECKING:`` block, whose body never runs."""
    test = node.test if isinstance(node, ast.If) else None
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            or isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _module_level_imports(tree):
    """The import nodes a module runs when it is imported: its body's, and those of top-level ``if``/``try`` blocks
    (not an ``if TYPE_CHECKING:`` body, which only a type checker reads)."""
    pending = list(tree.body)
    while pending:
        node = pending.pop(0)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.If, ast.Try)):
            pending += [*([] if _type_checking(node) else node.body), *node.orelse, *getattr(node, "finalbody", []),
                        *(child for handler in getattr(node, "handlers", []) for child in handler.body)]


def test_the_reference_package_imports_neither_torch_nor_genairr_at_module_level():
    """The reference is data and algorithms: a model file, a genotype or a user's FASTA reads into it without torch
    or the simulator; its only AlignAIR imports are the errors and itself."""
    root = ROOT / "reference"
    offending = []
    for path in sorted(root.rglob("*.py")):
        package = "alignair.reference" + "".join("." + part for part in path.relative_to(root).parts[:-1])
        for node in _module_level_imports(ast.parse(path.read_text())):
            names = ([importlib.util.resolve_name("." * node.level + (node.module or ""), package)] if
                     isinstance(node, ast.ImportFrom) and node.level else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names])
            for name in names:
                top = name.split(".")[0]
                if top in ("torch", "GenAIRR") or top == "alignair" and not name.startswith(
                        ("alignair.errors", "alignair.reference")):
                    offending.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
    assert offending == []
