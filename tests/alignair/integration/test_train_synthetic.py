import pytest
import torch
from torch.utils.data import DataLoader
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.losses.hierarchical import AlignAIRLoss
from alignair.data.experiment_presets import full_augmentation
from alignair.data.genairr import allele_vocab_from_dataconfig
from alignair.data.synthetic import SyntheticDataset
from alignair.data.collate import align_collate
from alignair.training.config import TrainingConfig
from alignair.training.trainer import Trainer


def test_train_few_steps_on_synthetic():
    torch.manual_seed(0)
    cfg_dc = gdata.HUMAN_IGH_OGRDB
    vocab = allele_vocab_from_dataconfig(cfg_dc)
    ds = SyntheticDataset(full_augmentation(cfg_dc), max_seq_length=576, has_d=True,
                          allele_vocab=vocab, n=16, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=align_collate)

    cfg = ModelConfig(max_seq_length=576, v_allele_count=len(vocab["V"]),
                      j_allele_count=len(vocab["J"]), d_allele_count=len(vocab["D"]),
                      has_d_gene=True)
    trainer = Trainer(SingleChainAlignAIR(cfg), AlignAIRLoss(cfg),
                      TrainingConfig(lr=1e-4, steps_per_epoch=3))
    logs = trainer.train_epoch(loader)
    assert torch.isfinite(torch.tensor(logs["loss"]))
