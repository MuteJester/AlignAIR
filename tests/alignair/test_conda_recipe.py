"""Validate the conda recipe without a conda toolchain: dependency invariants (so it can't
drift from pyproject) and that the Jinja renders to parseable YAML for both source modes."""
import tomllib
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
    assert f'set version = "{version}"' in META.read_text()


def test_recipe_smoke_commands_present():
    text = META.read_text()
    assert "alignair doctor" in text
    assert "alignair demo --steps 1" in text          # the "tiny demo" the review asked for
    assert "alignair = alignair.cli:main" in text     # entry point


def test_recipe_renders_for_both_source_modes():
    jinja2 = pytest.importorskip("jinja2")
    import yaml
    tmpl = jinja2.Environment().from_string(META.read_text())

    # default (release) mode -> PyPI url + sha placeholder/injected
    released = yaml.safe_load(tmpl.render(environ={}, PYTHON="$PYTHON"))
    assert "url" in released["source"] and "sha256" in released["source"]
    assert released["package"]["name"] == "alignair"

    # injected sha is honoured
    inj = yaml.safe_load(tmpl.render(environ={"ALIGNAIR_SDIST_SHA256": "a" * 64}, PYTHON="$PYTHON"))
    assert inj["source"]["sha256"] == "a" * 64

    # local-build mode -> path source (what CI uses to build/test without a published sdist)
    local = yaml.safe_load(tmpl.render(environ={"ALIGNAIR_LOCAL_BUILD": "1"}, PYTHON="$PYTHON"))
    assert local["source"].get("path") == ".."
    assert local["build"]["noarch"] == "python"
