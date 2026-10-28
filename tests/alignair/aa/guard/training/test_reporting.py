"""Guard summary publication preserves completed reports when replacement fails."""
import pytest

from alignair.aa.guard.training import reporting
from alignair.errors import OperationIOError
from alignair.io import atomic

pytestmark = pytest.mark.component


@pytest.fixture
def report_inputs():
    results = {
        'C1.s7': {'tag': 'C1.s7', 'cid': 'C1', 'seed': 7, 'status': 'ok', 'fold': 'igk_igl',
                  'dev_total_loss': .5, 'applicability_auroc': .8},
        'C1.s7.tcr': {'tag': 'C1.s7.tcr', 'cid': 'C1', 'seed': 7, 'status': 'ok', 'fold': 'tcr',
                      'dev_total_loss': .7, 'applicability_auroc': .9},
        'C2.s0': {'tag': 'C2.s0', 'cid': 'C2', 'status': 'engineering_failure',
                  'fold': 'tcr', 'step': 2, 'reason': 'injected failure'},
    }
    manifest = {'manifest_sha256': 'manifest', 'candidates': ['C1', 'C2'],
                'folds': {fold: {'n_train_after_exclusion': 10, 'n_excluded': 2} for fold in ('igk_igl', 'tcr')},
                'files_opened_audit': {'opened': ['development'], 'NOT_opened': ['challenge']}}
    return results, manifest


def _publish(kind, path, inputs):
    results, manifest = inputs
    if kind == 'sweep':
        reporting.write_sweep_report(results, {'C1': 7}, artifact_dir=path, manifest=manifest, device='cpu')
        return path/'STAGE4_RESULT.md'
    reporting.write_loco_report(results, artifact_dir=path, manifest=manifest, device='cpu')
    return path/'STAGE5_LOCO_TRAINING.md'


@pytest.mark.parametrize('kind', ['sweep', 'loco'])
def test_mixed_success_and_failure_report_is_utf8_and_includes_failed_runs(tmp_path, report_inputs, kind):
    path = _publish(kind, tmp_path/'nested', report_inputs)
    text = path.read_text(encoding='utf-8')
    assert 'C2.s0' in text and 'injected failure' in text and '2/3' in text
    assert text.endswith('\n') and 'No ' in text


@pytest.mark.parametrize('kind', ['sweep', 'loco'])
def test_replacement_failure_keeps_previous_report_and_reports_destination(tmp_path, monkeypatch, report_inputs, kind):
    path = _publish(kind, tmp_path, report_inputs); previous = path.read_bytes()
    def fail(*args): raise OSError('cannot replace summary')
    monkeypatch.setattr(atomic.os, 'replace', fail)
    with pytest.raises(OperationIOError) as caught: _publish(kind, tmp_path, report_inputs)
    assert caught.value.context == {'stage': 'guard_training_report', 'path': str(path)}
    assert isinstance(caught.value.__cause__, OSError)
    assert path.read_bytes() == previous and list(tmp_path.iterdir()) == [path]
