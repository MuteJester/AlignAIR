"""Tests for the training-diagnostics metrics."""
import torch

from alignair.config.alignair_config import AlignAIRConfig
from alignair.training import diagnostics as dg


def test_stable_rank_extremes():
    rank1 = torch.outer(torch.arange(1.0, 6), torch.arange(1.0, 4))     # rank-1
    assert abs(dg.stable_rank(rank1) - 1.0) < 1e-3
    ident = torch.eye(5)                                                # full rank
    assert abs(dg.stable_rank(ident) - 5.0) < 1e-3


def test_global_grad_norm():
    w = torch.nn.Parameter(torch.zeros(3))
    (w.sum() * 0 + (w * torch.tensor([3.0, 4.0, 0.0])).sum()).backward()
    assert abs(dg.global_grad_norm([w]) - 5.0) < 1e-5                   # ||[3,4,0]|| = 5


def test_eval_metrics():
    cfg = AlignAIRConfig(max_seq_length=64, v_allele_count=3, j_allele_count=2,
                         d_allele_count=2, has_d=True)
    out = {"v_allele": torch.tensor([[0.1, 0.9, 0.2]]), "d_allele": torch.tensor([[0.8, 0.1]]),
           "j_allele": torch.tensor([[0.2, 0.7]]),
           "mutation_rate": torch.tensor([[0.1]]), "indel_count": torch.tensor([[0.0]]),
           "productive": torch.tensor([[0.8]]),
           "orientation_logits": torch.tensor([[0.1, 2.0, 0.1, 0.1]])}
    for g in ("v", "d", "j"):
        out[f"{g}_start"] = torch.tensor([[10.0]]); out[f"{g}_end"] = torch.tensor([[100.0]])
    tgt = {"v_allele": torch.tensor([[0.0, 1.0, 0.0]]), "d_allele": torch.tensor([[1.0, 0.0]]),
           "j_allele": torch.tensor([[0.0, 1.0]]), "orientation": torch.tensor([1]),
           "mutation_rate": torch.tensor([[0.1]]), "indel_count": torch.tensor([[0.0]]),
           "productive": torch.tensor([[1.0]])}
    for g in ("v", "d", "j"):
        tgt[f"{g}_start"] = torch.tensor([[12.0]]); tgt[f"{g}_end"] = torch.tensor([[102.0]])
    m = dg.eval_metrics(out, tgt, cfg)
    assert m["v_allele_top1"] == 1.0 and m["d_allele_top1"] == 1.0 and m["j_allele_top1"] == 1.0
    assert m["orientation_acc"] == 1.0
    assert abs(m["v_seg_mae"] - 2.0) < 1e-5                             # (|10-12|+|100-102|)/2
    assert m["productive_acc"] == 1.0


def test_red_flags_detects_pathologies():
    metrics = {"grad_global": 5000.0, "numerical": {"nonfinite_loss": True},
               "modules": {"embedding": {"update_ratio": 0.5}},
               "activations": {"meta.proj": {"dead_frac": 0.8, "sat_frac": 0.1}},
               "weights": {"w": {"stable_rank": 1.0}}}
    flags = dg.red_flags(metrics)
    assert any("NON_FINITE" in f for f in flags)
    assert any("EXPLODING_GRAD" in f for f in flags)
    assert any("HOT_UPDATES" in f for f in flags)
    assert any("DEAD" in f for f in flags)
    assert any("RANK_COLLAPSE" in f for f in flags)


def test_activation_stats_on_model():
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=8, j_allele_count=4,
                         d_allele_count=4, has_d=True)
    from alignair.models.single_chain import SingleChainAlignAIR
    model = SingleChainAlignAIR(cfg)
    batch = {"tokenized_sequence": torch.randint(1, 6, (2, 256))}
    stats = dg.activation_stats(model, batch)
    assert "embedding" in stats
    for s in stats.values():
        assert 0.0 <= s["dead_frac"] <= 1.0 and 0.0 <= s["sat_frac"] <= 1.0


def test_monitor_is_pure_xray_no_interference():
    """The analytics must sit ABOVE the model: read-only, zero effect on training state."""
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=8, j_allele_count=4,
                         d_allele_count=4, has_d=True)
    from alignair.models.single_chain import SingleChainAlignAIR
    model = SingleChainAlignAIR(cfg)
    model.train()
    x = {"tokenized_sequence": torch.randint(1, 6, (2, 256))}
    model(x)                                            # train-mode forward -> populates BN running stats
    model(x)["v_allele"].sum().backward()               # populate grads

    bn = model.meta_tower.conv_layers[0].bn
    rm_before = bn.running_mean.clone()
    sd_before = {k: v.clone() for k, v in model.state_dict().items()}
    grad_before = model.embedding.token.weight.grad.clone()
    training_before = model.training

    dg.activation_stats(model, x)                       # the only stage that forwards the model

    assert model.training == training_before            # mode restored
    assert torch.equal(bn.running_mean, rm_before)      # eval-mode forward did NOT update BN stats
    assert torch.equal(model.embedding.token.weight.grad, grad_before)   # grads untouched
    for k, v in model.state_dict().items():
        assert torch.equal(v, sd_before[k]), k          # every weight/buffer unchanged
    assert all(len(m._forward_hooks) == 0 for m in model.modules())      # no lingering hooks
