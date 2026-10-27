import pytest
import torch

from alignair.core import AlignAIR, build_model_config
from alignair.core.boundaries import boundary_masks

pytestmark = pytest.mark.component


def _cfg(v=5, j=3):
    # max_seq_length must be >= 128 for the V/J classification tower (N=6 -> 1<<7).
    return build_model_config(v, j, input_space='protein', label_space='protein_classes')


def _tokens(lengths, L=192):
    t = torch.zeros(len(lengths), L, dtype=torch.long)
    for i, n in enumerate(lengths):
        t[i, :n] = torch.randint(1, 23, (n,))
    return t


def test_boundary_masks_start_is_0_to_n_minus_1_and_end_is_1_to_n():
    tokens = _tokens([8], L=16)
    sm, em = boundary_masks(tokens)
    assert sm[0, :8].all() and not sm[0, 8:].any()          # start slots 0..7
    assert not em[0, 0] and em[0, 1:9].all() and not em[0, 9:].any()   # end slots 1..8


def test_end_slot_n_is_reachable_by_the_decoder():
    """The rev-1 bug: slot n was masked out, so an exclusive end == n could never be decoded."""
    model = AlignAIR(_cfg()).eval()
    with torch.no_grad():
        head = model.branches['v'].end_head
        head.weight.zero_()
        head.bias.fill_(-1e4)
        head.bias[8] = 1e4
        output = model({'tokenized_sequence': _tokens([8])})
    assert output['v_end'].item() == pytest.approx(8., abs=1e-3)


def test_forward_emits_expected_keys_and_shapes_without_d():
    cfg = _cfg()
    out = AlignAIR(cfg).eval()({"tokenized_sequence": _tokens([120, 100])})
    for g, n_cls in (("v", 5), ("j", 3)):
        assert out[f"{g}_start_logits"].shape == (2, 192)
        assert out[f"{g}_end_logits"].shape == (2, 192)
        assert out[f"{g}_start"].shape == (2, 1) and out[f"{g}_end"].shape == (2, 1)
        assert out[f"{g}_class_logits"].shape == (2, n_cls)
    assert out["start_mask"].shape == (2, 192) and out["end_mask"].shape == (2, 192)
    assert not any(k.startswith("d_") for k in out)           # no D branch in Phase 1
    for absent in ("orientation_logits", "orientation", "mutation_rate", "indel_count", "productive"):
        assert absent not in out


def test_outputs_finite_for_short_but_nonempty_reads():
    out = AlignAIR(_cfg()).eval()({"tokenized_sequence": _tokens([1, 5])})
    for k, v in out.items():
        if v.dtype.is_floating_point:
            assert torch.isfinite(v).all(), k


def test_all_pad_row_is_rejected_not_silently_nan():
    with pytest.raises(ValueError, match="empty"):
        AlignAIR(_cfg()).eval()({"tokenized_sequence": torch.zeros(1, 192, dtype=torch.long)})


def test_model_builds_d_branch_when_has_d():
    cfg = build_model_config(5, 3, 4, input_space='protein', label_space='protein_classes', has_d=True)
    out = AlignAIR(cfg).eval()({"tokenized_sequence": _tokens([120, 100])})
    assert out["d_class_logits"].shape == (2, 4)              # D allele head over 4 D alleles
    assert out["d_start"].shape == (2, 1) and out["d_end"].shape == (2, 1)
