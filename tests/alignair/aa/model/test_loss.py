import pytest
import torch

from alignair.aa.loss import aa_loss, masked_soft_gaussian
from alignair.aa.model import boundary_masks

pytestmark = pytest.mark.component

L = 32


def _batch(n=8, B=2):
    tokens = torch.zeros(B, L, dtype=torch.long)
    tokens[:, :n] = 1
    return boundary_masks(tokens)


def _out(sm, em, vlogits, jlogits, seg_logits=None):
    B = vlogits.shape[0]
    sl = torch.zeros(B, L) if seg_logits is None else seg_logits
    o = {"start_mask": sm, "end_mask": em}
    for g in ("v", "j"):
        o[f"{g}_start_logits"], o[f"{g}_end_logits"] = sl.clone(), sl.clone()
        o[f"{g}_start"], o[f"{g}_end"] = torch.zeros(B, 1), torch.zeros(B, 1)
    o["v_class_logits"], o["j_class_logits"] = vlogits, jlogits
    return o


def _targets(B=2, n=8):
    return {"v_start": torch.zeros(B, 1), "v_end": torch.full((B, 1), 4.0),
            "j_start": torch.full((B, 1), 4.0), "j_end": torch.full((B, 1), float(n)),
            "v_class": torch.zeros(B, dtype=torch.long), "j_class": torch.zeros(B, dtype=torch.long)}


def test_masked_soft_gaussian_renormalizes_to_one_over_allowed_slots():
    _, em = _batch(n=8, B=1)
    t = masked_soft_gaussian(torch.tensor([[8.0]]), L, em)      # target at the terminal slot
    assert torch.allclose(t.sum(-1), torch.ones(1), atol=1e-6)  # renormalized
    assert float(t[0, 9:].sum()) == 0.0                          # no mass outside allowed slots
    assert float(t[0, 8]) > 0.3                                  # peak survives at the endpoint


def test_terminal_endpoint_is_learnable_and_correctly_rewarded():
    """The terminal exclusive end (== n) must be predictable AND rewarded.

    Note: this fixes one target and varies the PREDICTION. Comparing CE across two different
    targets would be invalid — a truncated edge Gaussian renormalizes to a lower entropy floor,
    so its CE is not comparable to a symmetric interior one.
    """
    from alignair.core.losses import _soft_ce
    _, em = _batch(n=8, B=1)
    tgt = masked_soft_gaussian(torch.tensor([[8.0]]), L, em)     # truth: exclusive end == n == 8

    def ce_when_model_says(k):
        lg = torch.zeros(1, L)
        lg[0, k] = 10.0
        return float(_soft_ce(lg, tgt, em))

    assert ce_when_model_says(8) < ce_when_model_says(7) < ce_when_model_says(6) \
        < ce_when_model_says(4)


def test_shared_mask_would_erase_the_terminal_target_rev1_regression():
    """rev-1 bug, pinned: with the nt-style single mask (slots 0..n-1) the true endpoint n gets
    ZERO target mass, making it unlearnable and silently under-penalized."""
    sm, em = _batch(n=8, B=1)
    assert not sm[0, 8] and em[0, 8]                              # start mask excludes n, end mask allows it
    lost = masked_soft_gaussian(torch.tensor([[8.0]]), L, sm)
    kept = masked_soft_gaussian(torch.tensor([[8.0]]), L, em)
    assert float(lost[0, 8]) == 0.0                              # the old behaviour: target erased
    assert float(kept[0, 8]) > 0.4                               # the fix: endpoint carries real mass


def test_loss_is_finite_and_components_are_means_not_sums():
    from alignair.core.losses import _soft_ce
    sm, em = _batch()
    out = _out(sm, em, torch.randn(2, 5), torch.randn(2, 3))
    tg = _targets()
    total, parts = aa_loss(out, tg)
    assert total.ndim == 0 and torch.isfinite(total)
    assert set(parts) == {"segmentation", "classification"}
    # explicitly recompute the 4 boundary terms and compare against their MEAN (a sum would be 4x).
    terms = []
    for g in ("v", "j"):
        for b, m in (("start", sm), ("end", em)):
            terms.append(_soft_ce(out[f"{g}_{b}_logits"], masked_soft_gaussian(tg[f"{g}_{b}"], L, m), m))
    expected_mean = torch.stack(terms).mean()
    assert torch.isclose(parts["segmentation"], expected_mean, atol=1e-6)
    assert not torch.isclose(parts["segmentation"], torch.stack(terms).sum(), atol=1e-6)


def test_confident_correct_class_gives_near_zero_classification():
    sm, em = _batch()
    v = torch.full((2, 5), -10.0)
    j = torch.full((2, 3), -10.0)
    v[:, 0] = 10.0
    j[:, 0] = 10.0
    _, parts = aa_loss(_out(sm, em, v, j), _targets())
    assert parts["classification"] < 1e-3


def test_no_d_keys_required():
    sm, em = _batch()
    total, _ = aa_loss(_out(sm, em, torch.randn(2, 5), torch.randn(2, 3)), _targets())
    assert torch.isfinite(total)
