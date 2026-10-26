import pytest
import json
import numpy as np
import pathlib
from alignair.aa.conformal import FrozenConformal

pytestmark = pytest.mark.component

FIX = json.loads((pathlib.Path(__file__).parent / "fixtures/conformal_fixture.json").read_text())


def test_predict_set_matches_hand_computed():
    fc = FrozenConformal(qhat=FIX["qhat"])
    for case in FIX["cases"]:
        assert sorted(fc.predict_set(np.array(case["probs"]))) == case["expected_set"]


def test_from_calibration_json_reads_full_group_quantile():
    cal = {"genes": {"V": {"calibration_frozen": {"q_mondrian": {"full": 0.5, "short": 0.6}}}}}
    assert FrozenConformal.from_calibration_json(cal, "V", "full").qhat == 0.5
