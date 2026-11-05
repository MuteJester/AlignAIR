"""Backbone transfer: shape-matched partial weight load."""
import pytest
import torch
import torch.nn as nn

from alignair.train.transfer import summarize, transfer_compatible_weights, load_source_state_dict

pytestmark = pytest.mark.integration


class _Net(nn.Module):
    def __init__(self, head_dim: int, shared_dim: int = 4):
        super().__init__()
        self.shared = nn.Linear(shared_dim, shared_dim)     # reference-agnostic (fixed shape)
        self.head = nn.Linear(shared_dim, head_dim)         # reference-specific (varies)


def test_transfer_copies_matching_shapes_only():
    src, tgt = _Net(head_dim=3), _Net(head_dim=5)
    with torch.no_grad():
        src.shared.weight.fill_(7.0)
        tgt.shared.weight.zero_()
    transferred, skipped = transfer_compatible_weights(tgt, src.state_dict())

    assert set(transferred) == {"shared.weight", "shared.bias"}
    assert set(skipped) == {"head.weight", "head.bias"}
    # the shared backbone was copied in place...
    assert torch.allclose(tgt.shared.weight, torch.full((4, 4), 7.0))
    # ...and the shape-mismatched head kept its own init (dim 5, not overwritten by src's dim 3)
    assert tgt.head.weight.shape[0] == 5


def test_transfer_ignores_extra_source_keys():
    tgt = _Net(head_dim=5)
    src_sd = dict(_Net(head_dim=5).state_dict())
    src_sd["nonexistent.weight"] = torch.zeros(2, 2)        # a key not in the target
    transferred, skipped = transfer_compatible_weights(tgt, src_sd)
    assert "nonexistent.weight" not in transferred and not skipped  # all target keys matched
    assert len(transferred) == len(tgt.state_dict())


def test_summarize_reports_percentage():
    s = summarize(["a.w", "a.b"], ["c.w"])
    assert "2/3" in s and "67%" in s


def test_safe_warm_start_decodes_weights_without_source_factories_or_rng_use(tmp_path, monkeypatch):
    import GenAIRR.data as gd
    from alignair import model_file
    from alignair.core import AlignAIR, prepare_model
    from alignair.core import components
    from alignair.reference import ReferenceSet
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), embed_dim=4, trunk_channels=8)
    model = setup.build_model()
    path = tmp_path / 'source.alignair'
    model_file.save_model(path, model, dataconfigs=[gd.HUMAN_IGH_OGRDB], training={},
                          label_schema=setup.label_schema(), include_trusted_pickle=False)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    monkeypatch.setattr(components, '_ARCHITECTURES', {})
    def forbidden(*args, **kwargs):
        pytest.fail('warm start constructed the source model')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    rng = torch.get_rng_state().clone()
    actual = load_source_state_dict(path)
    assert all(torch.equal(value, expected[key]) for key, value in actual.items())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('trust_pickle', [False, True])
def test_a_pt_warm_start_source_is_refused(tmp_path, trust_pickle):
    from alignair.errors import ModelError
    path = tmp_path / 'source.pt'
    torch.save({'model': _Net(3).state_dict()}, path)
    with pytest.raises(ModelError, match=r'AlignAIR 3 checkpoint .*alignair<4'):
        load_source_state_dict(path, trust_pickle=trust_pickle)
