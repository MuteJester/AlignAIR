"""Cohort content identity and split failures, independent of live simulation."""
from copy import deepcopy
from types import SimpleNamespace
import builtins
import os
import subprocess
import sys

import numpy as np
import pytest

from alignair.aa.convergence import cohort as module
from alignair.errors import ConfigurationError, InputError, MissingDependencyError, TrainingDataError

pytestmark = pytest.mark.component


def content():
    return {"schema": module.COHORT_SCHEMA, "seed": 1, "cal_frac": .3,
            "by_stratum": {"clean": [
                {"aa": "AAA", "v_call": "V0", "j_call": "J0", "d_call": "D0",
                 "d_inverted": False, "record": {"v_sequence_end": 3, "productive": True}},
                {"aa": "CAA", "v_call": "V1", "j_call": "J0", "d_call": "D0"}]},
            "calibration": [{"aa": "ACA", "v_call": "V0", "j_call": "J0", "d_call": "D0"}]}


@pytest.mark.parametrize("change", ["inversion", "boundary", "eligibility", "sequence", "label",
                                   "calibration", "seed", "fraction", "order", "stratum"])
def test_fingerprint_covers_metric_inputs_metadata_and_order(change):
    original = content()
    altered = deepcopy(original)
    row = altered["by_stratum"]["clean"][0]
    if change == "inversion": row["d_inverted"] = True
    elif change == "boundary": row["record"]["v_sequence_end"] = 6
    elif change == "eligibility": row["record"]["productive"] = False
    elif change == "sequence": row["aa"] = "ACC"
    elif change == "label": row["v_call"] = "V2"
    elif change == "calibration": altered["calibration"][0]["aa"] = "CCC"
    elif change == "seed": altered["seed"] = 2
    elif change == "fraction": altered["cal_frac"] = .2
    elif change == "order": altered["by_stratum"]["clean"].reverse()
    elif change == "stratum": altered["by_stratum"]["other"] = altered["by_stratum"].pop("clean")
    assert module.cohort_fingerprint(original) != module.cohort_fingerprint(altered)


def test_mapping_order_is_irrelevant_and_json_round_trip_keeps_identity():
    import json
    cohort = content()
    before = deepcopy(cohort)
    reordered = dict(reversed(list(cohort.items())))
    reordered["by_stratum"]["clean"][0] = dict(reversed(list(cohort["by_stratum"]["clean"][0].items())))
    assert module.cohort_fingerprint(cohort) == module.cohort_fingerprint(reordered)
    assert module.cohort_fingerprint(cohort) == module.cohort_fingerprint(json.loads(json.dumps(cohort)))
    assert cohort == before


@pytest.mark.parametrize("cohort", [None, {}, {"schema": "old"},
    {"schema": module.COHORT_SCHEMA, "value": float("nan")},
    {"schema": module.COHORT_SCHEMA, "value": object()}])
def test_unversioned_or_non_json_content_is_rejected(cohort):
    with pytest.raises(InputError) as caught:
        module.cohort_fingerprint(cohort)
    assert caught.value.context["stage"] == "convergence_cohort"


@pytest.mark.parametrize("values,field", [({"n_per_stratum": 0}, "n_per_stratum"),
    ({"n_per_stratum": True}, "n_per_stratum"), ({"seed": -1}, "seed"), ({"seed": 1.5}, "seed"),
    ({"cal_frac": 0}, "cal_frac"), ({"cal_frac": 1}, "cal_frac"),
    ({"cal_frac": float("nan")}, "cal_frac"), ({"cal_frac": True}, "cal_frac"),
    ({"cal_frac": ".3"}, "cal_frac"), ({"dc_name": None}, "dc_name"), ({"dc_name": ""}, "dc_name")])
def test_invalid_requests_fail_before_reference_construction(monkeypatch, values, field):
    monkeypatch.setattr(module, "build_protein_reference", lambda *_: pytest.fail("reference constructed"))
    with pytest.raises(ConfigurationError) as caught:
        module.build_cohort(**values)
    assert caught.value.context == {"stage": "convergence_cohort", "field": field}


@pytest.fixture
def cases(monkeypatch):
    import alignair_benchmark
    rows = []
    monkeypatch.setattr(alignair_benchmark, "stream_benchmark", lambda *args, **kwargs: iter(rows))
    maps = {gene: SimpleNamespace(allele_class={gene + "0": 0}) for gene in ("V", "J")}
    monkeypatch.setattr(module, "build_protein_reference", lambda *_: SimpleNamespace(class_maps=maps, db_hash="fixture"))
    return rows


def case(index):
    record = {"sequence": "GCT" * 20, "sequence_aa": "A" * 20, "junction_start": 12,
              "v_call": "V0", "j_call": "J0", "sequence_id": str(index)}
    return SimpleNamespace(record=record, orientation_id=0, stratum="clean_full")


def test_only_forward_cases_enter_the_cohort_even_when_reverse_translation_is_available(cases):
    cases.extend(case(index) for index in range(3))
    cases[1].orientation_id = 1
    cohort, _ = module.build_cohort()
    rows = cohort["calibration"] + cohort["by_stratum"]["clean"]
    assert {row["record"]["sequence_id"] for row in rows} == {"0", "2"}


def test_unknown_reference_calls_are_excluded_before_the_split(cases):
    cases.extend(case(index) for index in range(3))
    cases[-1].record["v_call"] = "QUARANTINED"
    cohort, _ = module.build_cohort()
    rows = cohort["calibration"] + cohort["by_stratum"]["clean"]
    assert {row["record"]["sequence_id"] for row in rows} == {"0", "1"}


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_clean_rows_fail_instead_of_returning_an_empty_evaluation(cases, count):
    cases.extend(case(index) for index in range(count))
    with pytest.raises(TrainingDataError) as caught:
        module.build_cohort()
    assert caught.value.context["clean_records"] == count


@pytest.mark.parametrize("fraction", [.001, .3, .999])
def test_small_valid_split_is_nonempty_disjoint_and_augmentation_uses_only_evaluation(cases, fraction):
    cases.extend(case(index) for index in range(2))
    state = np.random.get_state()
    cohort, fingerprint = module.build_cohort(cal_frac=fraction)
    assert len(cohort["calibration"]) == len(cohort["by_stratum"]["clean"]) == 1
    assert cohort["calibration"][0]["record"] is not cohort["by_stratum"]["clean"][0]["record"]
    for name, rows in cohort["by_stratum"].items():
        if name != "clean":
            assert all("record" not in row for row in rows)
    assert fingerprint == module.cohort_fingerprint(cohort)
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])


@pytest.mark.parametrize("dependency", ["alignair_benchmark", "nested_dependency"])
def test_missing_generation_package_is_contextual_but_nested_import_failures_are_preserved(monkeypatch, dependency):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "alignair_benchmark":
            raise ModuleNotFoundError("missing", name=dependency)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    expected = MissingDependencyError if dependency == "alignair_benchmark" else ModuleNotFoundError
    with pytest.raises(expected):
        module.build_cohort()


def test_checkpoint_scoring_imports_without_the_source_only_generation_package(repo_root):
    code = '''import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname == "alignair_benchmark" or fullname.startswith("alignair_benchmark."):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
from alignair.aa.convergence.evaluator import evaluate_checkpoint
assert callable(evaluate_checkpoint)
assert not any(name.startswith("alignair_benchmark") for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
                            env={**os.environ, "PYTHONPATH": str(repo_root / "src")})
    assert result.returncode == 0, result.stderr
