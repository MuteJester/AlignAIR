"""The README works on its own for a repertoire researcher: short enough to read, a first alignment near the top, the
seven model ids, install, the sections in the order a newcomer needs them, and nothing that belongs on the website."""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.component

README = Path(__file__).resolve().parents[2] / "README.md"
MODEL_IDS = ["alignair4-igh-human", "alignair4-igk-human", "alignair4-igl-human", "alignair4-tra-human",
             "alignair4-trb-human", "alignair4-trd-human", "alignair4-trg-human"]
SECTIONS = ["Install", "Check it works", "Your first alignment", "In Python", "Pretrained models", "Your own data",
            "How it compares", "Documentation", "Cite"]


def test_the_readme_is_short_and_runs_a_first_alignment_near_the_top():
    lines = README.read_text().splitlines()
    assert 150 <= len(lines) <= 260
    first_predict = next(i for i, line in enumerate(lines, 1) if line.lstrip("$ ").startswith("alignair predict"))
    assert first_predict <= 70
    assert any("pip install alignair" in line for line in lines)


def test_the_readme_names_the_seven_models():
    text = README.read_text()
    assert [model_id for model_id in MODEL_IDS if model_id not in text] == []


def test_the_readme_sections_come_in_a_newcomers_order():
    headings = [line[3:].strip() for line in README.read_text().splitlines() if line.startswith("## ")]
    assert [h for h in headings if h in SECTIONS] == SECTIONS


def test_the_readme_leaves_the_depth_to_the_website():
    text = README.read_text()
    assert not re.search(r"Sharing a trained model|Reproducible rows|Design (&|and) internals", text)
