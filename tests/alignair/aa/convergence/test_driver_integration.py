"""Reference-bound model construction for convergence checkpoint evaluation."""
import pytest

pytestmark = pytest.mark.integration


def test_candidate_reload_uses_shared_reference_bound_setup(tmp_path, monkeypatch):
    import torch
    from alignair.aa.convergence import driver
    from alignair.train.checkpoints import save_snapshot
    setup = driver._study_setup('HUMAN_IGH_OGRDB', 192)
    model = setup.build_model()
    path = tmp_path / 'candidate.pt'
    save_snapshot(path, model, setup.config, 7)
    calls = []
    original = driver._study_setup
    def prepare(name, length):
        calls.append((name, length))
        return original(name, length)
    monkeypatch.setattr(driver, '_study_setup', prepare)
    loaded, rebound = driver._load_fresh_candidate(str(path), 'HUMAN_IGH_OGRDB', 192, 'cpu')
    assert calls == [('HUMAN_IGH_OGRDB', 192)]
    assert rebound.label_schema() == setup.label_schema() and not loaded.training
    assert loaded is not model
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
