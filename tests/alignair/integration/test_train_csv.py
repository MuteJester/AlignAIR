import torch
from torch.utils.data import DataLoader
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.losses.hierarchical import AlignAIRLoss
from alignair.data.dataset import AlignAIRDataset
from alignair.data.collate import align_collate
from alignair.training.config import TrainingConfig
from alignair.training.trainer import Trainer

CSV = "tests/data/test/sample_igh.csv"


def test_train_on_sample_csv_loss_decreases(tmp_path):
    torch.manual_seed(0)
    ds = AlignAIRDataset(CSV, max_seq_length=576, has_d=True, nrows=16)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=align_collate)

    cfg = ModelConfig(max_seq_length=576, v_allele_count=ds.v_allele_count,
                      j_allele_count=ds.j_allele_count, d_allele_count=ds.d_allele_count,
                      has_d_gene=True)
    model = SingleChainAlignAIR(cfg)
    loss_fn = AlignAIRLoss(cfg)
    # lr=1e-4 is stable for this larger config (L=576, ~200 V classes); lr=1e-3
    # can spike transiently in the first ~20 steps before settling.
    trainer = Trainer(model, loss_fn, TrainingConfig(lr=1e-4, epochs=1))

    # One batch, many steps -> loss should fall.
    x, y = next(iter(loader))
    first = trainer.train_step(x, y)["loss"]
    for _ in range(30):
        last = trainer.train_step(x, y)["loss"]
    assert last < first

    # Save + resume restores weights exactly.
    ckpt = tmp_path / "c.pt"
    trainer.save_checkpoint(str(ckpt), epoch=1)
    model2 = SingleChainAlignAIR(cfg)
    trainer2 = Trainer(model2, AlignAIRLoss(cfg), TrainingConfig(lr=1e-3))
    assert trainer2.load_checkpoint(str(ckpt))["epoch"] == 1
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_fit_runs_one_epoch():
    ds = AlignAIRDataset(CSV, max_seq_length=576, has_d=True, nrows=8)
    loader = DataLoader(ds, batch_size=4, collate_fn=align_collate)
    cfg = ModelConfig(max_seq_length=576, v_allele_count=ds.v_allele_count,
                      j_allele_count=ds.j_allele_count, d_allele_count=ds.d_allele_count,
                      has_d_gene=True)
    trainer = Trainer(SingleChainAlignAIR(cfg), AlignAIRLoss(cfg),
                      TrainingConfig(epochs=1, steps_per_epoch=2))
    history = trainer.fit(loader)
    assert len(history) == 1 and "loss" in history[0]
