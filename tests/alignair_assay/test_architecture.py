"""The package's dependency rule, read from every module's syntax tree.

Each subpackage may import only the layers listed for it in ``ALLOWED``; a new subpackage needs a row before it
passes. ``alignair`` is imported only by ``core/reference.py``, ``alignair_benchmark`` never, and the pure layers
(truth, metrics, stats) do no file, process or network IO.
"""
import ast
from pathlib import Path

import alignair_assay

PACKAGE = Path(alignair_assay.__file__).resolve().parent
LAYERS = ("core", "truth", "airr", "metrics", "stats", "report", "scoring", "suites", "runners", "workflow", "cli")
ALL = set(LAYERS)
ALLOWED = {"core": {"core"}, "truth": {"core", "truth"}, "airr": {"core", "airr"},
           "metrics": {"core", "truth", "metrics"}, "stats": {"core", "metrics", "stats"},
           "report": {"core", "stats", "report"}, "scoring": {"core", "truth", "airr", "metrics", "stats", "scoring"},
           "suites": {"core", "suites"}, "runners": {"core", "airr", "truth", "suites", "runners"},
           "workflow": ALL - {"cli"}, "cli": ALL, "testing": ALL}
# Top-level modules that are not layers: the package marker and the ``python -m`` entry point.
ENTRY_POINTS = {"__init__": set(), "__main__": {"cli"}}
PURE = ("truth", "metrics", "stats")
IO_MODULES = {"subprocess", "argparse", "gzip", "csv", "shutil", "socket", "urllib", "http", "requests"}
IO_CALLS = {"open", "read_text", "write_text", "read_bytes", "write_bytes"}


def _modules():
    """``(owner, path relative to the package, syntax tree)`` per module; the owner is the module's subpackage, or
    its own name for a top-level module."""
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        owner = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        yield owner, rel, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(rel: Path, tree: ast.AST):
    """``(line, absolute dotted name)`` of every import in the tree, function-level ones included; ``from X import
    name`` gives ``X.name`` because the name may be a submodule."""
    package = ["alignair_assay", *rel.parts[:-1]]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module
            for alias in node.names:
                yield node.lineno, f"{module}.{alias.name}"


def test_layering():
    problems = []
    for owner, rel, tree in _modules():
        if len(rel.parts) == 1 and owner not in {*ENTRY_POINTS, "testing"}:
            problems.append(f"{rel}: top-level module; put it in a layer")
        elif len(rel.parts) > 1 and owner not in ALLOWED:
            problems.append(f"{rel}: subpackage {owner!r} has no row in ALLOWED")
        allowed = ALLOWED.get(owner, ENTRY_POINTS.get(owner, set()))
        for line, name in _imports(rel, tree):
            parts = name.split(".")
            if parts[0] == "alignair_assay" and len(parts) > 1 and parts[1] not in allowed:
                problems.append(f"{rel}:{line}: {owner} may not import alignair_assay.{parts[1]}")
    assert not problems, "\n".join(dict.fromkeys(problems))


def test_no_benchmark_import():
    found = [f"{rel}:{line}" for _, rel, tree in _modules() for line, name in _imports(rel, tree)
             if name.split(".")[0] == "alignair_benchmark"]
    assert not found, found


def test_alignair_imported_only_by_core_reference():
    importers = {str(rel) for _, rel, tree in _modules() for _, name in _imports(rel, tree)
                 if name.split(".")[0] == "alignair"}
    # The one import is inside a method, so this also shows function-level imports are seen.
    assert importers == {str(Path("core", "reference.py"))}


def test_pure_layers_do_no_io():
    problems = []
    for owner, rel, tree in _modules():
        if owner not in PURE:
            continue
        for line, name in _imports(rel, tree):
            if name.split(".")[0] in IO_MODULES:
                problems.append(f"{rel}:{line}: imports {name}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if called in IO_CALLS:
                    problems.append(f"{rel}:{node.lineno}: calls {called}()")
    assert not problems, "\n".join(problems)
