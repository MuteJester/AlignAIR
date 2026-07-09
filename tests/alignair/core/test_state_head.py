"""Per-position edit-state head: build, back-compat, loss wiring, learnability."""
import pytest
import torch
import torch.nn.functional as F

from alignair.core import AlignAIR
from alignair.core.config import AlignAIRConfig
from alignair.core.losses import make_logvars


def _cfg(**kw):
    return AlignAIRConfig(v_allele_count=5, j_allele_count=3, d_allele_count=2, has_d=True,
                          max_seq_length=128, **kw)


def test_state_head_off_by_default_no_logits():
    model = AlignAIR(_cfg())
    assert model.state_branch is None
    out = model({"tokenized_sequence": torch.randint(1, 6, (2, 128)),
                 "orientation": torch.zeros(2, dtype=torch.long)})
    assert "state_logits" not in out
    assert "state" not in make_logvars(_cfg())


def test_state_head_emits_per_position_logits_when_enabled():
    cfg = _cfg(state_head=True)
    model = AlignAIR(cfg)
    assert model.state_branch is not None
    out = model({"tokenized_sequence": torch.randint(1, 6, (2, 128)),
                 "orientation": torch.zeros(2, dtype=torch.long)})
    assert out["state_logits"].shape == (2, 128, 4)          # (B, L, |STATES|)
    assert "state" in make_logvars(cfg)


def test_loss_adds_state_term_and_ignores_padding():
    from alignair.core.losses import hierarchical_loss
    cfg = _cfg(state_head=True)
    model = AlignAIR(cfg)
    logvars = make_logvars(cfg)
    B, L = 2, 128
    out = model({"tokenized_sequence": torch.randint(1, 6, (B, L)),
                 "orientation": torch.zeros(B, dtype=torch.long)})
    genes = ["v", "d", "j"]
    targets = {}
    for g in genes:
        targets[f"{g}_start"] = torch.full((B, 1), 5.0)
        targets[f"{g}_end"] = torch.full((B, 1), 20.0)
        targets[f"{g}_allele"] = torch.zeros(B, getattr(cfg, f"{g}_allele_count"))
        targets[f"{g}_allele"][:, 0] = 1.0
    for k in ("mutation_rate", "indel_count", "productive"):
        targets[k] = torch.zeros(B, 1)
    labels = torch.randint(0, 4, (B, L))
    labels[:, 120:] = -100                                   # padded tail must be ignored
    targets["state_labels"] = labels
    total, parts = hierarchical_loss(out, targets, cfg, logvars)
    assert "state" in parts and torch.isfinite(parts["state"])


def test_state_branch_can_overfit_a_fixed_batch():
    """Learnability sanity: the tower + head must be able to fit per-position labels on a fixed batch."""
    torch.manual_seed(0)
    model = AlignAIR(_cfg(state_head=True))
    tokens = torch.randint(1, 6, (4, 128))
    inp = {"tokenized_sequence": tokens, "orientation": torch.zeros(4, dtype=torch.long)}
    labels = torch.randint(0, 4, (4, 128))
    opt = torch.optim.Adam(model.state_branch.parameters(), lr=1e-2)
    for _ in range(300):
        loss = F.cross_entropy(model(inp)["state_logits"].reshape(-1, 4), labels.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    acc = (model(inp)["state_logits"].argmax(-1) == labels).float().mean()
    assert acc > 0.9


@pytest.mark.slow
def test_train_step_end_to_end_includes_state_loss():
    import itertools
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.gym import Curriculum
    from alignair.train.trainer import _stream_records, build_batch, train_step

    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576, state_head=True)
    model = AlignAIR(cfg)
    logvars = make_logvars(cfg)
    recs = list(itertools.islice(_stream_records(gd.HUMAN_IGH_OGRDB, dict(Curriculum().params(0.6)), 0), 4))
    batch_in, targets = build_batch(recs, ref, cfg)
    assert targets["state_labels"].shape == (4, cfg.max_seq_length)
    opt = torch.optim.AdamW(list(model.parameters()) + list(logvars.parameters()), lr=1e-3)
    _, parts = train_step(model, batch_in, targets, cfg, logvars, opt)
    assert "state" in parts and parts["state"] == parts["state"]     # present and finite
