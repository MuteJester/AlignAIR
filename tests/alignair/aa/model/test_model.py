import pytest
import torch
import torch.nn.functional as F

from alignair.aa.model import AAAlignAIR, AAGeneBranch, boundary_masks
from alignair.core.config import AlignAIRConfig

pytestmark = pytest.mark.component


def _cfg(v=5, j=3):
    # max_seq_length must be >= 128 for the V/J classification tower (N=6 -> 1<<7).
    return AlignAIRConfig(v_allele_count=v, j_allele_count=j, d_allele_count=0,
                          has_d=False, num_chain_types=1, max_seq_length=192, vocab_size=23)


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
    tokens = _tokens([8], L=16)
    _, em = boundary_masks(tokens)
    logits = torch.full((1, 16), -1e4)
    logits[0, 8] = 1e4                                       # model wants to say "end = 8"
    dec = logits.masked_fill(~em, torch.finfo(logits.dtype).min)
    pos = torch.arange(16, dtype=torch.float32)
    assert abs(float((F.softmax(dec, -1) * pos).sum()) - 8.0) < 1e-3


def test_forward_emits_expected_keys_and_shapes_without_d():
    cfg = _cfg()
    out = AAAlignAIR(cfg).eval()({"tokenized_sequence": _tokens([120, 100])})
    for g, n_cls in (("v", 5), ("j", 3)):
        assert out[f"{g}_start_logits"].shape == (2, 192)
        assert out[f"{g}_end_logits"].shape == (2, 192)
        assert out[f"{g}_start"].shape == (2, 1) and out[f"{g}_end"].shape == (2, 1)
        assert out[f"{g}_class_logits"].shape == (2, n_cls)
    assert out["start_mask"].shape == (2, 192) and out["end_mask"].shape == (2, 192)
    assert not any(k.startswith("d_") for k in out)           # no D branch in Phase 1
    for absent in ("orientation_logits", "orientation", "mutation_rate", "indel_count", "productive"):
        assert absent not in out


def test_classify_returns_raw_linear_output_deterministically():
    """Deterministic check (not the flaky 'some logit is <0 or >1' heuristic)."""
    cfg = _cfg()
    model = AAAlignAIR(cfg).eval()
    br = model.branches["v"]
    tokens = _tokens([100])
    emb = model.embedding(tokens)
    sm, em = boundary_masks(tokens)
    _, _, s_exp, e_exp = br.segment_with_masks(emb, sm, em)
    with torch.no_grad():
        got = br.classify(emb, s_exp, e_exp)
        mask = br.cutout(s_exp, e_exp).unsqueeze(-1)
        want = br.cls_head(F.silu(br.cls_mid(br.cls_tower(emb * mask))))
    assert torch.allclose(got, want, atol=1e-6)


def test_outputs_finite_for_short_but_nonempty_reads():
    out = AAAlignAIR(_cfg()).eval()({"tokenized_sequence": _tokens([1, 5])})
    for k, v in out.items():
        if v.dtype.is_floating_point:
            assert torch.isfinite(v).all(), k


def test_all_pad_row_is_rejected_not_silently_nan():
    with pytest.raises(ValueError, match="empty"):
        AAAlignAIR(_cfg()).eval()({"tokenized_sequence": torch.zeros(1, 192, dtype=torch.long)})


def test_model_builds_d_branch_when_has_d():
    cfg = AlignAIRConfig(v_allele_count=5, j_allele_count=3, d_allele_count=4, has_d=True,
                         num_chain_types=1, max_seq_length=192, vocab_size=23)
    out = AAAlignAIR(cfg).eval()({"tokenized_sequence": _tokens([120, 100])})
    assert out["d_class_logits"].shape == (2, 4)              # D allele head over 4 D alleles
    assert out["d_start"].shape == (2, 1) and out["d_end"].shape == (2, 1)
