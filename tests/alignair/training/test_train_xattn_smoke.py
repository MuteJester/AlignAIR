import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner


def test_train_xattn_runs_and_checkpoints(tmp_path):
    import scripts.train_xattn as T
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=4, dim_feedforward=64)
    save = tmp_path / "xattn_smoke.pt"
    model = T.train_xattn(cfg, gdata.HUMAN_IGK_OGRDB, steps=5, batch_size=8, lr=1e-3,
                          device="cpu", save=str(save), ckpt_every=5, eval_n=0, progress=False)
    assert save.exists()
    ck = torch.load(str(save), map_location="cpu", weights_only=False)
    m2 = XAttnAligner(DNAlignAIRConfig(**ck["config"]))
    m2.load_state_dict(ck["model"])                       # round-trips into a fresh model
    p1 = dict(model.named_parameters())
    for n, p in m2.named_parameters():
        assert torch.allclose(p, p1[n].cpu(), atol=1e-5)
