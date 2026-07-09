"""Segmentation pad-masking: boundary logits must be restricted to the valid read length so the
soft-argmax can't be pulled into the right-padding (the short-read coordinate bug)."""
import torch

from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair.core.losses import hierarchical_loss, make_logvars


def _cfg():
    return AlignAIRConfig(max_seq_length=256, v_allele_count=8, j_allele_count=4,
                          d_allele_count=4, has_d=True)


def _padded_batch(cfg, B=4, n=60):
    """B reads of true length n in a max_seq_length window -> lots of right padding."""
    tok = torch.zeros(B, cfg.max_seq_length, dtype=torch.long)
    tok[:, :n] = torch.randint(1, 6, (B, n))
    return {"tokenized_sequence": tok, "orientation": torch.zeros(B, dtype=torch.long)}


def test_position_mask_marks_only_the_read():
    cfg = _cfg(); n = 60
    out = AlignAIR(cfg).eval()(_padded_batch(cfg, n=n))
    assert "position_mask" in out
    assert out["position_mask"][:, :n].all() and not out["position_mask"][:, n:].any()


def test_boundary_expectation_stays_inside_the_read():
    """The soft-argmax must not land in the pad region (positions >= read length)."""
    cfg = _cfg(); n = 60
    out = AlignAIR(cfg).eval()(_padded_batch(cfg, n=n))
    for g in ("v", "d", "j"):
        for b in ("start", "end"):
            e = out[f"{g}_{b}"].squeeze(-1)
            assert (e <= n).all(), f"{g}_{b} expectation {e.tolist()} leaked past read length {n}"


def test_no_padding_mask_is_noop():
    cfg = _cfg()
    tok = torch.randint(1, 6, (2, cfg.max_seq_length))          # no zeros -> nothing to mask
    out = AlignAIR(cfg).eval()({"tokenized_sequence": tok})
    assert out["position_mask"].all()


def test_hierarchical_loss_finite_with_padding():
    """Masking to -inf on pad must not produce 0*-inf NaN in the soft-CE."""
    cfg = _cfg(); B, n = 4, 60
    model = AlignAIR(cfg)
    batch = _padded_batch(cfg, B=B, n=n)
    tg = {"mutation_rate": torch.rand(B, 1), "indel_count": torch.zeros(B, 1),
          "productive": torch.ones(B, 1), "orientation": torch.zeros(B, dtype=torch.long)}
    for g, cnt in (("v", 8), ("j", 4), ("d", 4)):
        tg[f"{g}_start"] = torch.full((B, 1), 10.0); tg[f"{g}_end"] = torch.full((B, 1), 40.0)
        y = torch.zeros(B, cnt); y[:, 0] = 1.0; tg[f"{g}_allele"] = y
    total, parts = hierarchical_loss(model(batch), tg, cfg, make_logvars(cfg))
    assert torch.isfinite(total), parts
    total.backward()                                            # gradients must be finite too
    assert torch.isfinite(model.branches["v"].start_head.weight.grad).all()
