import logging
import pytest
import torch
from torch.utils.data import DataLoader
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.gym.gym import AlignAIRGym
from alignair.gym.collate import gym_collate


def test_gym_streams_batches_with_full_targets(caplog):
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    gym = AlignAIRGym([gdata.HUMAN_IGH_OGRDB], rs, n=12, seed=0, log_every=4)
    with caplog.at_level(logging.INFO):
        gym.set_progress(0.5)   # mid curriculum -> should log the stage
    loader = DataLoader(gym, batch_size=4,
                        collate_fn=lambda b: gym_collate(b, rs, has_d=True))
    batch = next(iter(loader))
    B = batch["tokens"].shape[0]
    assert B == 4
    assert batch["mask"].dtype == torch.bool
    assert batch["region_labels"].shape == batch["tokens"].shape
    assert batch["state_labels"].shape == batch["tokens"].shape
    assert batch["v_allele"].shape == (B, len(rs.gene("V").names))
    assert batch["v_germline_start"].shape == (B,)
    assert batch["productive"].shape == (B, 1)
    # verbose curriculum line was logged
    assert any("curriculum stage" in m.lower() for m in caplog.messages)


def test_gym_difficulty_affects_generation():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    gym = AlignAIRGym([gdata.HUMAN_IGH_OGRDB], rs, n=20, seed=1)
    gym.set_progress(0.0)
    clean = list(gym)
    gym.set_progress(1.0)
    hard = list(gym)
    # harder curriculum yields more sequencing noise on average
    avg_clean = sum(t["noise_count"] for t in clean) / len(clean)
    avg_hard = sum(t["noise_count"] for t in hard) / len(hard)
    assert avg_hard >= avg_clean
