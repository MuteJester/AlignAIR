"""The GenAIRR series AlignAIR is built and tested on is the one the package and the conda recipe declare."""
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.component


def test_genairr_is_pinned_to_the_2_4_series():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "GenAIRR>=2.4.0,<2.5" in project["dependencies"]
    recipe = [line.split("#")[0].strip() for line in (ROOT / "conda" / "meta.yaml").read_text().splitlines()]
    assert "- genairr >=2.4.0,<2.5" in recipe
