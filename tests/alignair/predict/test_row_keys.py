"""The rows ``Aligner.predict`` returns keep their keys in today's order (a user who writes them with ``json.dumps``
or iterates them sees it): the light records, the AIRR rows, and the rows of reads whose D is short, against the key
lists recorded at 38d72394 (``fixtures/test_chip_row_keys.json``). The golden check sorts keys, so only this test
catches a reorder."""
import json
from pathlib import Path

import pytest

from tests.alignair.predict.golden_rows import FIXTURE, golden_model
from tests.alignair.predict.test_d_gate import DPresenceStub, _encoded

pytestmark = pytest.mark.integration

ROW_KEYS = Path(__file__).parents[1] / "fixtures" / "test_chip_row_keys.json"


@pytest.fixture(scope="module")
def golden():
    return golden_model()


def test_rows_keep_their_key_order(golden):
    from alignair import Aligner
    model, reference = golden
    reads = json.loads(FIXTURE.read_text())["reads"]
    keys = json.loads(ROW_KEYS.read_text())
    light = Aligner.from_model(model, reference, device="cpu").predict(reads, batch_size=8, airr=False).records
    assert [list(row) for row in light] == keys["light"]
    rows = Aligner.from_model(model, reference, device="cpu").predict(reads, batch_size=8).records
    assert [list(row) for row in rows] == keys["airr"]
    short = DPresenceStub(model, {_encoded(read): 1 for read in reads[:5]})
    rows = Aligner.from_model(short, reference, device="cpu").predict(reads[:5], batch_size=8).records
    assert [list(row) for row in rows] == keys["short_d"]
