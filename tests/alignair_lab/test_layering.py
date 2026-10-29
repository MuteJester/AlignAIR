"""The lab's dependency rule, read from every module's syntax tree.

Each lab module may import only the lab modules listed for it in ``ALLOWED``; a new module needs a row before it
passes. ``architectures`` holds candidate model code shaped like the product's, so it imports the product and nothing
of the assay or of the rest of the lab. No lab module imports ``alignair_benchmark``.
"""
import ast
from pathlib import Path

import pytest

import alignair_lab

pytestmark = pytest.mark.component

PACKAGE = Path(alignair_lab.__file__).resolve().parent
MODULES = ("recipe", "records", "headline", "architectures", "screen", "cli")
ALLOWED = {"recipe": {"recipe"}, "records": {"records"}, "headline": {"headline"},
           "architectures": {"architectures"},
           "screen": {"recipe", "records", "headline", "architectures", "screen"},
           "cli": set(MODULES), "__main__": {"cli"}, "__init__": set()}


def _owner(rel: Path) -> str:
    """The module a file belongs to: its subpackage, or its own name for a top-level module."""
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        yield _owner(rel), rel, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(rel: Path, tree: ast.AST):
    """``(line, absolute dotted name)`` of every import, function-level ones included; ``from X import name`` gives
    ``X.name``, since the name may be a submodule."""
    package = ["alignair_lab", *rel.parts[:-1]]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = ".".join(package[:len(package) - node.level + 1] + ([node.module] if node.module else [])) \
                if node.level else node.module
            for alias in node.names:
                yield node.lineno, f"{module}.{alias.name}"


def _problems(modules) -> list[str]:
    problems = []
    for owner, rel, tree in modules:
        if owner not in ALLOWED:
            problems.append(f"{rel}: module {owner!r} has no row in ALLOWED")
        for line, name in _imports(rel, tree):
            parts = name.split(".")
            if parts[0] == "alignair_benchmark":
                problems.append(f"{rel}:{line}: imports alignair_benchmark")
            elif parts[0] == "alignair_assay" and owner == "architectures":
                problems.append(f"{rel}:{line}: architectures may not import the assay")
            elif parts[0] == "alignair_lab" and len(parts) > 1 and parts[1] not in ALLOWED.get(owner, set()):
                problems.append(f"{rel}:{line}: {owner} may not import alignair_lab.{parts[1]}")
    return problems


def _synthetic(*modules):
    return [(_owner(Path(rel)), Path(rel), ast.parse(source)) for rel, source in modules]


def test_layering():
    problems = _problems(_modules())
    assert not problems, "\n".join(problems)


def test_the_rule_refuses_what_it_names():
    assert _problems(_synthetic(("screen.py", "from alignair_lab.records import CachedRecordSource"),
                                ("cli.py", "from . import screen"),
                                ("architectures/null.py", "from alignair.core import register_architecture"))) == []
    refused = {("records.py", "from alignair_lab.screen import run_screen"): "records may not import "
                                                                           "alignair_lab.screen",
               ("architectures/wide.py", "from alignair_assay.core import layout"): "may not import the assay",
               ("architectures/wide.py", "from ..records import write_record_cache"): "architectures may not import "
                                                                                     "alignair_lab.records",
               ("headline.py", "import alignair_benchmark"): "imports alignair_benchmark",
               ("sweep.py", ""): "module 'sweep' has no row in ALLOWED"}
    for module, problem in refused.items():
        found = _problems(_synthetic(module))
        assert any(problem in text for text in found), (module, found)
