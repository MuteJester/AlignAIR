import json

import pytest

from alignair.applicability.training import budget
from alignair.errors import ArtifactIntegrityError, InputError, NumericalError, OperationIOError

pytestmark = pytest.mark.component


def test_plan_counts_every_seed_replicate_and_model():
    e = budget.plan(["C1", "C2", "C3", "C4"], seed_replicates=3, loco_folds=4)
    assert e.main_seed_runs == 12 and e.loco_models == 16
    assert e.total_models == 12 + 4 + 16 + 1 + 4          # main-seeds + chosen + loco + C0 + C0-withheld


def test_record_and_total_roundtrip(tmp_path):
    p = tmp_path / "b.json"
    budget.record("C1.s0", 100, 0.5, 1000, path=p)
    budget.record("C1.s1", 200, 1.0, 1000, path=str(p))
    t = budget.total(path=p)
    assert t["wall_s"] == 300 and t["gpu_h"] == 1.5 and t["n_runs"] == 2


def test_resumed_run_replaces_its_totals_and_preserves_other_records(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({
        "study": "guard", "runs": {"C1.s0": {"wall_s": 10, "gpu_h": 0.1, "params": 100}},
    }))
    budget.record("C2.s0", 20, 0.2, 200, path=path)
    budget.record("C1.s0", 30, 0.3, 100, path=path)
    assert budget.total(path=path) == {"wall_s": 50, "gpu_h": 0.5, "n_runs": 2}
    assert json.loads(path.read_text())["study"] == "guard"


def test_missing_ledger_has_zero_totals_without_creating_a_file(tmp_path):
    path = tmp_path / "missing.json"
    assert budget.total(path=path) == {"wall_s": 0, "gpu_h": 0, "n_runs": 0}
    assert not path.exists()


def test_malformed_ledger_is_not_overwritten(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text("incomplete ledger")
    with pytest.raises(ArtifactIntegrityError) as caught:
        budget.record("C1.s0", 1, 0.1, 100, path=path)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)
    assert path.read_text() == "incomplete ledger"


@pytest.mark.parametrize('changes', [{'model_id': ''}, {'model_id': 1}, {'wall_s': -.1},
    {'wall_s': True}, {'wall_s': '1'}, {'wall_s': 2**10000}, {'wall_s': float('nan')},
    {'gpu_h': float('inf')}, {'gpu_h': -1}, {'params': 1.9}, {'params': True}, {'params': -1}])
def test_invalid_record_never_creates_or_replaces_a_ledger(tmp_path, changes):
    path = tmp_path/'ledger.json'
    values = dict(model_id='C1.s0', wall_s=1., gpu_h=.1, params=100, path=path); values.update(changes)
    with pytest.raises(InputError): budget.record(**values)
    assert not path.exists()
    path.write_bytes(b'existing work')
    with pytest.raises(InputError): budget.record(**values)
    assert path.read_bytes() == b'existing work'


@pytest.mark.parametrize('data', [[], {}, {'runs': []}, {'runs': {'': {}}}, {'runs': {'id': []}},
    {'runs': {'id': {'wall_s': '1', 'gpu_h': 0, 'params': 1}}},
    {'runs': {'id': {'wall_s': -1, 'gpu_h': 0, 'params': 1}}},
    {'runs': {'id': {'wall_s': 1, 'gpu_h': 0, 'params': 1.5}}},
    {'runs': {'id': {'wall_s': 1, 'params': 1}}},
    {'runs': {}, 'metadata': float('nan')}])
def test_malformed_existing_ledger_is_refused_by_both_read_and_update(tmp_path, data):
    path = tmp_path/'ledger.json'; path.write_text(json.dumps(data)); previous = path.read_bytes()
    for operation in (lambda: budget.total(path=path), lambda: budget.record('new', 1, .1, 100, path=path)):
        with pytest.raises(ArtifactIntegrityError) as caught: operation()
        assert caught.value.context['path'] == str(path)
        assert path.read_bytes() == previous


def test_invalid_utf8_ledger_is_not_replaced(tmp_path):
    path = tmp_path/'ledger.json'; path.write_bytes(b'\xff')
    with pytest.raises(ArtifactIntegrityError) as caught: budget.record('new', 1, 0, 1, path=path)
    assert isinstance(caught.value.__cause__, UnicodeError) and path.read_bytes() == b'\xff'


def test_json_exponent_overflow_in_existing_metadata_is_not_silently_rewritten(tmp_path):
    path = tmp_path/'ledger.json'; path.write_text('{"runs": {}, "metadata": 1e999}')
    previous = path.read_bytes()
    with pytest.raises(ArtifactIntegrityError): budget.record('new', 1, 0, 1, path=path)
    assert path.read_bytes() == previous


def test_unreadable_ledger_reports_io_error_with_cause(tmp_path):
    with pytest.raises(OperationIOError) as caught: budget.total(path=tmp_path)
    assert caught.value.context['path'] == str(tmp_path) and isinstance(caught.value.__cause__, OSError)


def test_atomic_ledger_replacement_failure_preserves_all_previous_runs(tmp_path, monkeypatch):
    from alignair.io import atomic
    path = tmp_path/'budget.json'; budget.record('first', 10, .5, 100, path=path); previous = path.read_bytes()
    def fail(*args): raise OSError('disk unavailable')
    monkeypatch.setattr(atomic.os, 'replace', fail)
    with pytest.raises(OperationIOError): budget.record('second', 20, .5, 100, path=path)
    assert path.read_bytes() == previous and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('value', [1e308, 10**308])
def test_overflowing_aggregate_is_a_numerical_error(tmp_path, value):
    path = tmp_path/'budget.json'
    path.write_text(json.dumps({'runs': {str(i): {'wall_s': value, 'gpu_h': 0, 'params': 1} for i in range(2)}}))
    with pytest.raises(NumericalError): budget.total(path=path)


def test_nested_ledger_parent_is_created_for_a_valid_record(tmp_path):
    path = tmp_path/'nested'/'budget.json'
    budget.record('run', 1, 0, 1, path=path)
    assert budget.total(path=path) == {'wall_s': 1., 'gpu_h': 0., 'n_runs': 1}
