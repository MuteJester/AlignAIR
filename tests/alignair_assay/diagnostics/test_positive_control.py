"""The readiness audit's positive control (``workflow.audit.base``): zeroing ``POSITIVE_MODULE`` in a trained chip
must move the boundary ``POSITIVE_METRIC`` scores, or the control cannot make its metric worse."""
import json
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

from alignair.model_file import load_model  # noqa: E402
from tests.support.boundaries import decoded_boundaries  # noqa: E402
from alignair_assay.diagnostics.perturb import zero_module  # noqa: E402
from alignair_assay.workflow.audit.base import POSITIVE_METRIC, POSITIVE_MODULE  # noqa: E402

pytestmark = pytest.mark.integration

TESTS = Path(__file__).resolve().parents[2]
TEST_CHIP = TESTS / "alignair" / "fixtures" / "test_chip.alignair"
READS = json.loads((TESTS / "alignair" / "predict" / "fixtures" / "golden_reads.json").read_text())["reads"]


def _boundaries(path, gene, edge):
    from alignair.aligner import Aligner
    loaded = load_model(str(path))
    outputs = Aligner(loaded.model.eval(), loaded.reference).predict_raw(READS, batch_size=len(READS))
    start, end = decoded_boundaries(outputs, gene)
    return (start if edge == "start" else end).tolist()


def test_zeroing_the_positive_module_moves_the_boundary_its_metric_scores(tmp_path):
    _, gene, name, decoder = POSITIVE_METRIC.split(".")             # head.<gene>.<edge>_exact.<decoder>
    edge = name.removesuffix("_exact")
    assert decoder == "argmax" and edge in ("start", "end")
    zeroed = tmp_path / "zeroed.alignair"
    zero_module(TEST_CHIP, zeroed, POSITIVE_MODULE)
    before, after = _boundaries(TEST_CHIP, gene, edge), _boundaries(zeroed, gene, edge)
    moved = sum(a != b for a, b in zip(before, after))
    assert moved >= len(READS) * 0.6, (POSITIVE_METRIC, moved, len(READS))
