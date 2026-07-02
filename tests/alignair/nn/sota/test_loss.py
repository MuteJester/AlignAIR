"""Detector loss: 1-D GIoU, presence-masking, and that it trains toward the target."""
import torch

from alignair.nn.sota.loss import DetectorLoss, interval_giou
from alignair.nn.sota.query_decoder import GENES


def test_interval_giou_identical_is_one_and_disjoint_is_negative():
    ident = interval_giou(torch.tensor([[0.2, 0.6]]), torch.tensor([[0.2, 0.6]]))
    assert torch.allclose(ident, torch.ones(1), atol=1e-5)
    disjoint = interval_giou(torch.tensor([[0.0, 0.1]]), torch.tensor([[0.9, 1.0]]))
    assert disjoint.item() < 0                          # no overlap + far apart -> negative GIoU


def test_interval_giou_is_order_agnostic():
    a = interval_giou(torch.tensor([[0.6, 0.2]]), torch.tensor([[0.2, 0.6]]))   # pred reversed
    assert torch.allclose(a, torch.ones(1), atol=1e-5)


def _out_and_targets(B=4, K=6):
    out, tgt = {}, {}
    for g in GENES:
        out[g] = {"span": torch.rand(B, 2), "objectness": torch.randn(B),
                  "allele_scores": torch.randn(B, K), "trim": torch.rand(B, 2)}
        allele = torch.zeros(B, K); allele[:, 0] = 1.0
        tgt[g] = {"span": torch.rand(B, 2), "present": torch.ones(B), "allele": allele,
                  "trim": torch.rand(B, 2)}
    return out, tgt


def test_loss_is_scalar_and_logs_all_terms():
    out, tgt = _out_and_targets()
    loss, logs = DetectorLoss()(out, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)
    for g in GENES:
        for term in ("span", "giou", "obj", "allele", "trim"):
            assert f"{g}/{term}" in logs
    assert "total" in logs


def test_absent_gene_only_supervises_objectness():
    """When a gene is absent (present=0), span/trim/allele must not contribute — only objectness."""
    out, tgt = _out_and_targets()
    for g in GENES:
        tgt[g]["present"] = torch.zeros_like(tgt[g]["present"])
        tgt[g]["allele"] = torch.zeros_like(tgt[g]["allele"])       # no positive -> zero allele loss
    _, logs = DetectorLoss()(out, tgt)
    for g in GENES:
        assert logs[f"{g}/span"] == 0.0 and logs[f"{g}/trim"] == 0.0 and logs[f"{g}/allele"] == 0.0
        assert logs[f"{g}/obj"] > 0.0


def test_loss_decreases_as_prediction_approaches_target():
    torch.manual_seed(0)
    B, K = 8, 5
    tgt = {}
    for g in GENES:
        allele = torch.zeros(B, K); allele[:, 2] = 1.0
        tgt[g] = {"span": torch.full((B, 2), 0.5), "present": torch.ones(B), "allele": allele,
                  "trim": torch.full((B, 2), 0.3)}
    crit = DetectorLoss()

    def loss_for(scale):
        out = {}
        for g in GENES:
            scores = torch.full((B, K), -2.0); scores[:, 2] = scale        # confident on true allele
            out[g] = {"span": torch.full((B, 2), 0.5), "objectness": torch.full((B,), scale),
                      "allele_scores": scores, "trim": torch.full((B, 2), 0.3)}
        return crit(out, tgt)[0]

    assert loss_for(5.0) < loss_for(-1.0)              # confident+correct beats unconfident
