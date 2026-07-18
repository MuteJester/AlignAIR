"""Validate the conda recipe without a conda toolchain: dependency invariants (so it can't
drift from pyproject) and that the Jinja renders to parseable YAML for both source modes."""
try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "conda" / "meta.yaml"


def _dep_lines():
    """The `- pkg ...` lines under requirements (raw, Jinja-free section)."""
    text = META.read_text()
    return [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]


def test_recipe_deps_match_the_cli_dependency_story():
    deps = " ".join(_dep_lines())
    # #93 removed typer/rich; the recipe must not resurrect them
    assert "typer" not in deps and "rich" not in deps
    # the [cli] extra is reflected
    for pkg in ("pytorch", "genairr", "argcomplete", "parasail-python", "airr", "huggingface_hub"):
        assert pkg in deps, f"{pkg} missing from recipe run deps"


def test_recipe_version_matches_pyproject():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert f'version: "{version}"' in META.read_text()


def test_recipe_smoke_commands_present():
    text = META.read_text()
    assert "alignair doctor" in text
    assert "alignair demo --steps 1" in text          # the "tiny demo" the review asked for
    assert "alignair = alignair.cli:main" in text     # entry point


def test_recipe_is_literal_local_source_and_renders_stably():
    """The recipe is literal / local-source (no top-level Jinja vars, no environ), so it renders the
    same regardless of context - which is what makes conda-build's two-pass render robust."""
    jinja2 = pytest.importorskip("jinja2")
    import yaml
    tmpl = jinja2.Environment().from_string(META.read_text())
    rendered = yaml.safe_load(tmpl.render(environ={}, PYTHON="$PYTHON"))
    assert rendered["package"]["name"] == "alignair"
    assert str(rendered["package"]["version"]) == "3.0.0"
    assert rendered["source"].get("path") == ".."          # local source, no PyPI url / sha
    assert "url" not in rendered["source"]
    assert rendered["build"]["noarch"] == "python"
    # rendering does not depend on environ (so both conda-build passes agree)
    again = yaml.safe_load(tmpl.render(environ={"ALIGNAIR_LOCAL_BUILD": "", "X": "y"}, PYTHON="$PYTHON"))
    assert again == rendered
