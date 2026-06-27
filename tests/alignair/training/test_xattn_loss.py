import random
import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.reference.reference_set import ReferenceSet
from alignair.core.xattn_aligner import XAttnAligner
from alignair.training.reader import build_sibling_index
from alignair.training.xattn_loss import xattn_losses
from alignair.gym import AlignAIRGym, gym_collate
from torch.utils.data import DataLoader


def _batch_and_model():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = XAttnAligner(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=4, dim_feedforward=64))
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=8, seed=0)
    loader = DataLoader(gym, batch_size=8, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    batch = next(iter(loader))
    return model, rs, batch


def test_losses_finite_and_have_parts():
    model, rs, batch = _batch_and_model()
    ref_emb = model.encode_reference(rs)
    sib = build_sibling_index(rs)
    total, parts = xattn_losses(model, batch, ref_emb, sib, random.Random(0))
    assert torch.isfinite(total)
    for k in ("orientation", "region", "allele", "gstart", "gend"):
        assert k in parts and torch.isfinite(parts[k])


def test_loss_is_differentiable():
    model, rs, batch = _batch_and_model()
    ref_emb = model.encode_reference(rs)
    sib = build_sibling_index(rs)
    total, _ = xattn_losses(model, batch, ref_emb, sib, random.Random(0))
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
