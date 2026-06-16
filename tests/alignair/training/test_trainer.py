import torch
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.losses.hierarchical import AlignAIRLoss
from alignair.training.config import TrainingConfig
from alignair.training.trainer import Trainer


def _tiny_model_and_loss():
    cfg = ModelConfig(max_seq_length=256, v_allele_count=4, j_allele_count=3,
                      d_allele_count=4, has_d_gene=True)
    return SingleChainAlignAIR(cfg), AlignAIRLoss(cfg), cfg


def _one_batch(cfg, B=4):
    L = cfg.max_seq_length
    x = {"tokenized_sequence": torch.randint(0, 6, (B, L))}
    y = {
        "v_start": torch.full((B, 1), 1.0), "v_end": torch.full((B, 1), 100.0),
        "j_start": torch.full((B, 1), 120.0), "j_end": torch.full((B, 1), 200.0),
        "d_start": torch.full((B, 1), 105.0), "d_end": torch.full((B, 1), 110.0),
        "v_allele": torch.zeros(B, 4), "j_allele": torch.zeros(B, 3), "d_allele": torch.zeros(B, 4),
        "mutation_rate": torch.full((B, 1), 0.1), "indel_count": torch.full((B, 1), 1.0),
        "productive": torch.ones(B, 1),
    }
    y["v_allele"][:, 0] = 1.0; y["j_allele"][:, 0] = 1.0; y["d_allele"][:, 0] = 1.0
    return x, y


def test_single_train_step_returns_finite_loss():
    model, loss_fn, cfg = _tiny_model_and_loss()
    trainer = Trainer(model, loss_fn, TrainingConfig(lr=1e-3))
    x, y = _one_batch(cfg)
    logs = trainer.train_step(x, y)
    assert torch.isfinite(torch.tensor(logs["loss"]))


def test_overfits_single_batch():
    torch.manual_seed(0)
    model, loss_fn, cfg = _tiny_model_and_loss()
    trainer = Trainer(model, loss_fn, TrainingConfig(lr=1e-3))
    x, y = _one_batch(cfg)
    first = trainer.train_step(x, y)["loss"]
    for _ in range(15):
        last = trainer.train_step(x, y)["loss"]
    assert last < first  # loss decreases on a repeated batch


def test_checkpoint_save_and_resume(tmp_path):
    model, loss_fn, cfg = _tiny_model_and_loss()
    trainer = Trainer(model, loss_fn, TrainingConfig(lr=1e-3))
    x, y = _one_batch(cfg)
    trainer.train_step(x, y)
    ckpt = tmp_path / "ck.pt"
    trainer.save_checkpoint(str(ckpt), epoch=1)

    model2, loss2, _ = _tiny_model_and_loss()
    trainer2 = Trainer(model2, loss2, TrainingConfig(lr=1e-3))
    state = trainer2.load_checkpoint(str(ckpt))
    assert state["epoch"] == 1
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
