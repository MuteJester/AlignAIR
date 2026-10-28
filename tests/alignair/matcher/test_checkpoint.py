
import pytest
import torch
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch
from alignair.matcher.checkpoint import save_checkpoint, load_checkpoint, read_checkpoint
from alignair.errors import ConfigurationError, ModelFormatError, ModelIntegrityError, OperationIOError, TrainingError

pytestmark = pytest.mark.component


def _tiny():
    return MatcherConfig(d_model=16, d_match=8, n_layers=2, dilations=(1,))


def test_roundtrip_preserves_step_weights_and_optimizer(tmp_path):
    cfg = _tiny()
    m = AIRRMatch(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # one update so the optimizer accumulates state (momentum/variance)
    loss = sum((p ** 2).sum() for p in m.parameters())
    loss.backward()
    opt.step()
    path = tmp_path / "ck.pt"
    save_checkpoint(path, cfg, m, opt, step=1234)

    m2 = AIRRMatch(cfg)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    step = load_checkpoint(path, m2, opt2, device="cpu")

    assert step == 1234
    assert len(opt2.state) > 0                                   # optimizer state restored
    assert torch.equal(next(m.parameters()), next(m2.parameters()))  # weights match


def test_weights_only_warmstart_ignores_optimizer(tmp_path):
    cfg = _tiny()
    m = AIRRMatch(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, cfg, m, opt, step=500)

    m2 = AIRRMatch(cfg)
    step = load_checkpoint(path, m2, opt=None, device="cpu")     # warm-start, no optimizer
    assert step == 500
    assert torch.equal(next(m.parameters()), next(m2.parameters()))


@pytest.mark.parametrize('state', [[], {}, {'model': {}, 'config': {}, 'step': True},
                                 {'model': {}, 'config': {}, 'step': -1}, {'model': [], 'config': {}}])
def test_malformed_data_only_checkpoint_is_classified(tmp_path, state):
    path = tmp_path / 'bad.pt'
    torch.save(state, path)
    with pytest.raises(ModelFormatError):
        read_checkpoint(path)


def test_reader_rejects_unrestricted_pickle_and_missing_files(tmp_path):
    path = tmp_path / 'bad.pt'
    torch.save({'object': object()}, path)
    with pytest.raises(ModelFormatError) as caught:
        read_checkpoint(path)
    assert caught.value.context['path'] == str(path)
    with pytest.raises(OperationIOError):
        read_checkpoint(tmp_path / 'missing.pt')


def test_failed_publication_keeps_previous_checkpoint_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'previous')
    def fail(state, temporary):
        from pathlib import Path
        Path(temporary).write_bytes(b'partial')
        raise OSError('disk error')
    monkeypatch.setattr(torch, 'save', fail)
    cfg = _tiny()
    model = AIRRMatch(cfg)
    with pytest.raises(OperationIOError):
        save_checkpoint(path, cfg, model, torch.optim.AdamW(model.parameters()), 1)
    assert path.read_bytes() == b'previous'
    assert sorted(item.name for item in tmp_path.iterdir()) == ['checkpoint.pt']


def test_resume_rejects_configuration_mismatch_before_model_mutation(tmp_path):
    from dataclasses import replace
    cfg = _tiny()
    source, target = AIRRMatch(cfg), AIRRMatch(replace(cfg, tau=.2))
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, cfg, source, torch.optim.AdamW(source.parameters()), 1)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(ConfigurationError):
        load_checkpoint(path, target, torch.optim.AdamW(target.parameters()))
    assert all(torch.equal(value, before[name]) for name, value in target.state_dict().items())
    assert load_checkpoint(path, target) == 1  # Explicit weight-only warm starts can change tau.


@pytest.mark.parametrize('optimizer', [None, {}])
def test_resume_requires_usable_optimizer_state(tmp_path, optimizer):
    cfg = _tiny()
    model = AIRRMatch(cfg)
    from dataclasses import asdict
    path = tmp_path / 'checkpoint.pt'
    torch.save({'config': asdict(cfg), 'model': model.state_dict(), 'opt': optimizer}, path)
    with pytest.raises((ModelFormatError, TrainingError)):
        load_checkpoint(path, model, torch.optim.AdamW(model.parameters()))


def test_strict_matcher_restore_uses_shared_weight_errors(tmp_path):
    from dataclasses import asdict
    cfg = _tiny()
    model = AIRRMatch(cfg)
    path = tmp_path / 'checkpoint.pt'
    torch.save({'config': asdict(cfg), 'model': {}}, path)
    with pytest.raises(ModelIntegrityError):
        load_checkpoint(path, model)
