import pytest
import torch

from alignair.aa.guard.fusion import FusionContractError, HmmFusion, fit_fusion

pytestmark = pytest.mark.component

NAMES = ["a", "b", "c"]
CK = "bank-checksum-abc"


def test_normalize_uses_stored_stats():
    f = HmmFusion(NAMES, [1, 2, 3], [1, 1, 1], CK)
    out = f.normalize([2, 4, 6], feature_names=NAMES, bank_checksum=CK)
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]))


def test_fuse_preserves_raw_vector_and_width():
    f = HmmFusion(NAMES, [0, 0, 0], [1, 1, 1], CK)
    seq = torch.zeros(1, 10)
    fused, raw = f.fuse(seq, [5, 6, 7], feature_names=NAMES, bank_checksum=CK)
    assert fused.shape == (1, 13)                       # seq_rep(10) + hmm(3)
    assert torch.allclose(raw, torch.tensor([5.0, 6.0, 7.0]))   # raw unchanged provenance


def test_feature_ordering_change_hard_fails_fusion():
    f = HmmFusion(NAMES, [0, 0, 0], [1, 1, 1], CK)
    with pytest.raises(FusionContractError):
        f.fuse(torch.zeros(1, 4), [1, 2, 3], feature_names=["a", "c", "b"], bank_checksum=CK)


def test_bank_checksum_change_hard_fails_fusion():
    f = HmmFusion(NAMES, [0, 0, 0], [1, 1, 1], CK)
    with pytest.raises(FusionContractError):
        f.fuse(torch.zeros(1, 4), [1, 2, 3], feature_names=NAMES, bank_checksum="changed")


def test_fit_fusion_roundtrip():
    f = fit_fusion(NAMES, [[0, 0, 0], [2, 2, 2]], CK)
    out = f.normalize([1, 1, 1], feature_names=NAMES, bank_checksum=CK)
    assert torch.allclose(out, torch.zeros(3))         # mean=1 -> normalized 0
