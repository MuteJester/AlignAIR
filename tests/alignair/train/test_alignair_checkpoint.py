import pytest
import GenAIRR.data as gd
import torch
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair.train.checkpoints import save_checkpoint, load_training_checkpoint
from alignair import model_file as mf

pytestmark = pytest.mark.integration


def test_trainer_save_checkpoint_writes_alignair(tmp_path):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    model = AlignAIR(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    p = tmp_path / "m.alignair"
    save_checkpoint(str(p), cfg, model, step=1000, opt=opt,
                    dataconfigs=[gd.HUMAN_IGH_OGRDB], train_args={"lr": 1e-4, "batch_size": 64})
    assert mf.container.is_alignair_file(str(p))
    md = mf.read_metadata(str(p))
    assert md["training"]["train_args"]["batch_size"] == 64
    ts = load_training_checkpoint(p)
    assert ts.step == 1000 and ts.optimizer_state is not None
    assert "python" in ts.rng and "torch" in ts.rng
