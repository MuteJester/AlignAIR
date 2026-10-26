"""The package's dependency rule, read from every module's syntax tree.

Each subpackage may import only the layers listed for it in ``ALLOWED``; a new subpackage needs a row before it
passes. ``real`` is the one nested package: each of its modules is a sub-layer of its own, ``real/<sub>.py`` owned
by ``real.<sub>`` (``real/__init__.py`` by ``real``), with its own row, and an import of
``alignair_assay.real.<sub>`` is an import of that sub-layer. ``workflow`` and ``cli`` may import any of them;
``report`` none (it renders plain dicts). ``alignair`` is imported only by ``core/reference.py`` and
``diagnostics/``, ``alignair_benchmark`` never, and the pure layers (truth, metrics, stats, real.truth,
real.metrics) do no file, process or network IO.
"""
import ast
from pathlib import Path

import alignair_assay

PACKAGE = Path(alignair_assay.__file__).resolve().parent
# Subpackages whose modules are each a sub-layer of their own.
NESTED = ("real",)
REAL = ("real", "real.fetch", "real.locus", "real.preprocess", "real.genotype", "real.truth", "real.metrics")
LAYERS = ("core", "truth", "airr", "metrics", "stats", "report", "scoring", "suites", "runners", "diagnostics",
          "workflow", "cli", *REAL)
ALL = set(LAYERS)
ALLOWED = {"core": {"core"}, "truth": {"core", "truth"}, "airr": {"core", "airr"},
           "metrics": {"core", "truth", "metrics"}, "stats": {"core", "metrics", "stats"},
           "report": {"core", "stats", "report"}, "scoring": {"core", "truth", "airr", "metrics", "stats", "scoring"},
           "suites": {"core", "suites"}, "runners": {"core", "airr", "truth", "suites", "runners"},
           "diagnostics": {"core", "truth", "airr", "metrics", "stats", "scoring", "suites", "diagnostics"},
           "workflow": ALL - {"cli"}, "cli": ALL, "testing": ALL,
           # The package marker imports nothing; each module is its own sub-layer.
           "real": set(),
           "real.genotype": {"core", "real.genotype"},
           "real.truth": {"core", "real.genotype", "real.truth"},
           "real.metrics": {"core", "metrics", "real.genotype", "real.truth", "real.metrics"},
           "real.fetch": {"core", "real.fetch"},
           "real.locus": {"core", "real.genotype", "real.locus"},
           "real.preprocess": {"core", "real.preprocess"}}
# Top-level modules that are not layers: the package marker and the ``python -m`` entry point.
ENTRY_POINTS = {"__init__": set(), "__main__": {"cli"}}
PURE = ("truth", "metrics", "stats", "real.truth", "real.metrics")
IO_MODULES = {"subprocess", "argparse", "gzip", "csv", "shutil", "socket", "urllib", "http", "requests"}
IO_CALLS = {"open", "read_text", "write_text", "read_bytes", "write_bytes"}


def _owner(rel: Path) -> str:
    """The layer a module belongs to: its subpackage, its own name for a top-level module, and ``<package>.<name>``
    for a module of a ``NESTED`` package (the package itself for its ``__init__.py``)."""
    if len(rel.parts) == 1:
        return rel.stem
    if rel.parts[0] in NESTED:
        if len(rel.parts) == 2 and rel.stem == "__init__":
            return rel.parts[0]
        return f"{rel.parts[0]}.{Path(rel.parts[1]).stem}"
    return rel.parts[0]


def _layer(name: str) -> str | None:
    """The layer an absolute dotted import of ``alignair_assay`` names (``alignair_assay.real.truth.X`` ->
    ``real.truth``); None for an import outside the package or of the package itself."""
    parts = name.split(".")
    if parts[0] != "alignair_assay" or len(parts) < 2:
        return None
    if parts[1] in NESTED and len(parts) > 2:
        return f"{parts[1]}.{parts[2]}"
    return parts[1]


def _modules():
    """``(owner, path relative to the package, syntax tree)`` per module (``_owner``)."""
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        yield _owner(rel), rel, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


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


def _layering_problems(modules) -> list[str]:
    problems = []
    for owner, rel, tree in modules:
        if len(rel.parts) == 1 and owner not in {*ENTRY_POINTS, "testing"}:
            problems.append(f"{rel}: top-level module; put it in a layer")
        elif len(rel.parts) > 1 and owner not in ALLOWED:
            problems.append(f"{rel}: subpackage {owner!r} has no row in ALLOWED")
        allowed = ALLOWED.get(owner, ENTRY_POINTS.get(owner, set()))
        for line, name in _imports(rel, tree):
            layer = _layer(name)
            if layer is not None and layer not in allowed:
                problems.append(f"{rel}:{line}: {owner} may not import alignair_assay.{layer}")
    return list(dict.fromkeys(problems))


def _io_problems(modules) -> list[str]:
    problems = []
    for owner, rel, tree in modules:
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
    return problems


def _synthetic(*modules: tuple[str, str]):
    """``(owner, rel, tree)`` for each ``(path relative to the package, source)``, as ``_modules`` gives them."""
    return [(_owner(Path(rel)), Path(rel), ast.parse(source)) for rel, source in modules]


def test_layering():
    problems = _layering_problems(_modules())
    assert not problems, "\n".join(problems)


def test_real_modules_are_owned_by_their_own_sublayer():
    assert _owner(Path("real", "genotype.py")) == "real.genotype"
    assert _owner(Path("real", "__init__.py")) == "real"
    assert _owner(Path("core", "reference.py")) == "core"
    assert _owner(Path("suites", "catalog", "s1.py")) == "suites"
    assert _owner(Path("testing.py")) == "testing"
    assert _layer("alignair_assay.real.truth.RealTruth") == "real.truth"
    assert _layer("alignair_assay.real.genotype") == "real.genotype"
    assert _layer("alignair_assay.real") == "real"
    assert _layer("alignair_assay.core.reference.Reference") == "core"
    assert _layer("alignair_assay") is None and _layer("json") is None


def test_real_sublayers_follow_their_rows():
    allowed = _synthetic(
        ("real/truth.py", "from alignair_assay.real.genotype import RealAllele\nfrom . import genotype"),
        ("real/metrics.py", "from alignair_assay.metrics import is_informational\nfrom .truth import RealTruth"),
        ("real/locus.py", "from alignair_assay.real.genotype import RealGenotype"),
        ("real/genotype.py", "from alignair_assay.core.reference import Reference"),
        ("workflow/real.py", "from alignair_assay.real import fetch, preprocess\nimport alignair_assay.real.metrics"),
        ("cli/real.py", "from alignair_assay.real.genotype import load_vdjbase_csv"),
        ("real/__init__.py", '"""marker"""'))
    assert _layering_problems(allowed) == []
    refused = {
        ("real/truth.py", "from alignair_assay.real.fetch import fetch_reads"): "real.truth may not import "
                                                                                 "alignair_assay.real.fetch",
        ("real/genotype.py", "from alignair_assay.metrics import registry"): "real.genotype may not import "
                                                                             "alignair_assay.metrics",
        ("real/locus.py", "from .truth import RealTruth"): "real.locus may not import alignair_assay.real.truth",
        ("report/real.py", "from alignair_assay.real.genotype import RealGenotype"): "report may not import "
                                                                                     "alignair_assay.real.genotype",
        ("real/fetch.py", "from alignair_assay.workflow import real"): "real.fetch may not import "
                                                                       "alignair_assay.workflow",
        ("real/novel.py", ""): "subpackage 'real.novel' has no row in ALLOWED",
    }
    for module, problem in refused.items():
        found = _layering_problems(_synthetic(module))
        assert any(problem in text for text in found), (module, found)


def test_real_truth_and_metrics_are_pure():
    io = _synthetic(("real/truth.py", "import csv"), ("real/metrics.py", "def f(p):\n    return p.read_text()"))
    assert len(_io_problems(io)) == 2
    assert _io_problems(_synthetic(("real/genotype.py", "import csv\nopen('x')"))) == []


def test_no_benchmark_import():
    found = [f"{rel}:{line}" for _, rel, tree in _modules() for line, name in _imports(rel, tree)
             if name.split(".")[0] == "alignair_benchmark"]
    assert not found, found


def _alignair_importers(modules):
    return {str(rel) for _, rel, tree in modules for _, name in _imports(rel, tree)
            if name.split(".")[0] == "alignair"}


def _may_import_alignair(rel: Path) -> bool:
    return rel == Path("core", "reference.py") or rel.parts[0] == "diagnostics"


def test_alignair_imported_only_by_core_reference_and_diagnostics():
    importers = _alignair_importers(_modules())
    assert all(_may_import_alignair(Path(name)) for name in importers), sorted(importers)
    # The core import is inside a method, so this also shows function-level imports are seen.
    assert str(Path("core", "reference.py")) in importers


def test_diagnostics_may_import_alignair():
    def module(owner, name, source):
        return owner, Path(owner, name), ast.parse(source)

    for source in ("import alignair", "from alignair.core import model", "def f():\n    import alignair.cli"):
        allowed = _alignair_importers([module("diagnostics", "replay.py", source)])
        assert allowed == {str(Path("diagnostics", "replay.py"))}
        assert all(_may_import_alignair(Path(name)) for name in allowed)
        refused = _alignair_importers([module("runners", "alignair.py", source)])
        assert refused and not any(_may_import_alignair(Path(name)) for name in refused)


def test_pure_layers_do_no_io():
    problems = _io_problems(_modules())
    assert not problems, "\n".join(problems)
