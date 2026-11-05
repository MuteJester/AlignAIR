"""The chip is the product's network, ``chip-v1``: built from ``alignair.network`` and ``alignair.refine`` with the
trunk width the configuration gives, and importing it loads neither the lab nor the assay."""
import subprocess
import sys

import pytest

from alignair.core import prepare_model

from .conftest import toy_reference

pytestmark = pytest.mark.component


def test_chip_v1_builds_the_chip_with_the_configured_trunk_width():
    from alignair.network.chip import ChipCore
    setup = prepare_model(toy_reference(), architecture="chip-v1", trunk_channels=8, max_seq_length=64, anchor_heads=True, d_presence_head=True)
    model = setup.build_model()
    assert isinstance(model.core, ChipCore) and model.core.trunk.project.out_channels == 8
    assert model.reference_agnostic


def test_the_product_network_imports_neither_the_lab_nor_the_assay():
    code = ("import sys, alignair.network.chip, alignair.network.objective, alignair.refine.vrescore; "
            "print(sorted({m.split('.')[0] for m in sys.modules if m.startswith(('alignair_lab', 'alignair_assay'))}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
