"""Tests for the reusable ModelXRay network-health analytics."""
import torch

from alignair.config.alignair_config import AlignAIRConfig
from alignair.training.alignair_trainer import eval_metrics
from alignair.xray import ModelXRay, network


def test_stable_rank_extremes():
    rank1 = torch.outer(torch.arange(1.0, 6), torch.arange(1.0, 4))
    assert abs(network.stable_rank(rank1) - 1.0) < 1e-3
    assert abs(network.stable_rank(torch.eye(5)) - 5.0) < 1e-3


def test_global_grad_norm():
    w = torch.nn.Parameter(torch.zeros(3))
    (w * torch.tensor([3.0, 4.0, 0.0])).sum().backward()
    assert abs(network.global_grad_norm([w]) - 5.0) < 1e-5


def test_eval_metrics():
    cfg = AlignAIRConfig(max_seq_length=64, v_allele_count=3, j_allele_count=2,
                         d_allele_count=2, has_d=True)
    out = {"v_allele": torch.tensor([[0.1, 0.9, 0.2]]), "d_allele": torch.tensor([[0.8, 0.1]]),
           "j_allele": torch.tensor([[0.2, 0.7]]), "mutation_rate": torch.tensor([[0.1]]),
           "indel_count": torch.tensor([[0.0]]), "productive": torch.tensor([[0.8]]),
           "orientation_logits": torch.tensor([[0.1, 2.0, 0.1, 0.1]])}
    tgt = {"v_allele": torch.tensor([[0.0, 1.0, 0.0]]), "d_allele": torch.tensor([[1.0, 0.0]]),
           "j_allele": torch.tensor([[0.0, 1.0]]), "orientation": torch.tensor([1]),
           "mutation_rate": torch.tensor([[0.1]]), "indel_count": torch.tensor([[0.0]]),
           "productive": torch.tensor([[1.0]])}
    for g in ("v", "d", "j"):
        out[f"{g}_start"] = torch.tensor([[10.0]]); out[f"{g}_end"] = torch.tensor([[100.0]])
        tgt[f"{g}_start"] = torch.tensor([[12.0]]); tgt[f"{g}_end"] = torch.tensor([[102.0]])
    m = eval_metrics(out, tgt, cfg)
    assert m["v_allele_top1"] == 1.0 and m["orientation_acc"] == 1.0
    assert abs(m["v_seg_mae"] - 2.0) < 1e-5 and m["productive_acc"] == 1.0


def test_red_flags_detects_pathologies():
    flags = network.red_flags({"grad_global": 5000.0, "numerical": {"nonfinite_loss": True},
                               "modules": {"embedding": {"update_ratio": 0.5}},
                               "activations": {"x": {"dead_frac": 0.8, "sat_frac": 0.1, "max_abs": 2}},
                               "weights": {"w": {"stable_rank": 1.0}}})
    for tag in ("NON_FINITE", "EXPLODING_GRAD", "HOT_UPDATES", "DEAD", "RANK_COLLAPSE"):
        assert any(tag in f for f in flags)


def _model():
    from alignair.models.single_chain import SingleChainAlignAIR
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=8, j_allele_count=4,
                         d_allele_count=4, has_d=True)
    return SingleChainAlignAIR(cfg), cfg


def test_model_xray_observe_records_health():
    from alignair.models.losses import make_logvars
    model, cfg = _model()
    logvars = make_logvars(cfg)
    xr = ModelXRay(model, lr=3e-4, deep_every=1, uncertainty=logvars)
    x = {"tokenized_sequence": torch.randint(1, 6, (2, 256))}
    model(x)["v_allele"].sum().backward()
    rec = xr.observe(step=1, loss=1.23, parts={"seg": 0.5}, probe_input=x)
    assert "grad_global" in rec and "modules" in rec and "uncertainty" in rec
    assert "weights" in rec and "activations" in rec           # deep (deep_every=1)
    assert isinstance(rec["flags"], list)


def test_model_xray_deep_pass_geometry_interference_and_report():
    from alignair.models.losses import hierarchical_loss, make_logvars
    model, cfg = _model()
    logvars = make_logvars(cfg)
    B = 4
    x = {"tokenized_sequence": torch.randint(1, 6, (B, 256)),
         "orientation": torch.zeros(B, dtype=torch.long)}
    tg = {"mutation_rate": torch.rand(B, 1), "indel_count": torch.zeros(B, 1),
          "productive": torch.ones(B, 1), "orientation": torch.zeros(B, dtype=torch.long)}
    for g, cnt in (("v", 8), ("j", 4), ("d", 4)):
        tg[f"{g}_start"] = torch.full((B, 1), 10.0); tg[f"{g}_end"] = torch.full((B, 1), 100.0)
        y = torch.zeros(B, cnt); y[:, 0] = 1.0; tg[f"{g}_allele"] = y

    def task_losses(m, inp):
        return hierarchical_loss(m(inp), tg, cfg, logvars)[1]

    xr = ModelXRay(model, lr=3e-4, deep_every=1, uncertainty=logvars,
                   task_losses=task_losses, shared_module="embedding")
    model(x)["v_allele"].sum().backward()
    rec = xr.observe(1, 1.0, probe_input=x)
    assert "geometry" in rec and "embedding" in rec["geometry"]         # per-layer feature geometry
    assert "interference" in rec and "min_cosine" in rec["interference"]  # cross-task gradient conflict
    assert "velocity" not in rec                                        # first deep pass -> no prev snapshot
    rec2 = xr.observe(2, 1.0, probe_input=x)
    assert "velocity" in rec2                                           # second deep pass -> velocity

    rep = xr.deep_report(probe_input=x)
    assert "weightwatcher_alpha" in rep and "cka" in rep


def test_model_xray_is_pure_xray_no_interference():
    """ModelXRay must sit ABOVE the model: zero mutation of weights/grads/mode/BN-stats/hooks."""
    model, _ = _model()
    model.train()
    x = {"tokenized_sequence": torch.randint(1, 6, (2, 256))}
    model(x)                                                    # populate BN running stats (train mode)
    model(x)["v_allele"].sum().backward()                       # populate grads
    bn = model.meta_tower.conv_layers[0].bn
    rm, sd = bn.running_mean.clone(), {k: v.clone() for k, v in model.state_dict().items()}
    grad = model.embedding.token.weight.grad.clone()
    was_training = model.training

    ModelXRay(model, lr=3e-4, deep_every=1).observe(1, 1.0, probe_input=x)

    assert model.training == was_training and torch.equal(bn.running_mean, rm)
    assert torch.equal(model.embedding.token.weight.grad, grad)
    for k, v in model.state_dict().items():
        assert torch.equal(v, sd[k]), k
    assert all(len(m._forward_hooks) == 0 for m in model.modules())
