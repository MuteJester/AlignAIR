"""Invalid study requests stop before checkpoint loading or any output side effect."""
import pytest

from alignair.aa.convergence import driver
from alignair.errors import ConfigurationError

pytestmark = pytest.mark.component


@pytest.mark.parametrize('changes', [{'budget': 0}, {'budget': True}, {'budget': 1.5},
    {'ckpt_steps': (1.9,)}, {'ckpt_steps': (True,)}, {'ckpt_steps': (0,)}, {'ckpt_steps': (3,)},
    {'ckpt_steps': (2,1)}, {'ckpt_steps': (1,1)}, {'ckpt_steps': '12'}, {'ckpt_steps': None},
    {'ckpt_steps': {1,2}}, {'batch_size': 0}, {'cohort_seed': -1}, {'n_per_stratum': 0},
    {'val_every': -1}, {'plateau_k': True}, {'base_lr': 0}, {'base_lr': True},
    {'base_lr': float('nan')}, {'base_lr': 2**10000}, {'lr_min': -.1}, {'lr_min': .01},
    {'lr_min': float('inf')}, {'device': 'bad-device'}, {'device': None}, {'device': 0},
    {'dc_name': ''}, {'resume_path': None}, {'resume_path': b'source.pt'}])
def test_invalid_request_cannot_load_allocate_or_create_outputs(tmp_path, monkeypatch, changes):
    def forbidden(*args, **kwargs): pytest.fail('checkpoint/model/cohort work preceded request validation')
    monkeypatch.setattr(driver, 'load_checkpoint', forbidden)
    monkeypatch.setattr(driver, 'build_cohort', forbidden)
    options = dict(resume_path='source.pt', budget=2, ckpt_steps=(1,2), device='cpu',
                   art_dir=tmp_path/'art', ckpt_dir=tmp_path/'ckpt')
    options.update(changes)
    with pytest.raises(ConfigurationError): driver.run_study(**options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('offsets, expected', [((), ()), (iter((1, 2)), (1, 2))])
def test_offsets_allow_empty_schedule_or_an_ordered_iterator(offsets, expected):
    from alignair.aa.convergence.validation import validate_study_request
    assert validate_study_request(2, offsets, 0, 1, 1, .001, 0, 0, 0, 'ref', 'cpu') == expected
