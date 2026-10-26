import pytest

from alignair.aa.guard.checkpoint import build_provenance, verify_provenance
from alignair.aa.guard.fusion import fit_fusion

pytestmark = pytest.mark.component


def _prov():
    return build_provenance(bank_checksum="BC", training_manifest_hash="TM", partition_hash="PH",
                            seed=7, loss_mask_config={"oe_cap": 0.15})


def test_provenance_binds_and_verifies():
    p = _prov()
    d = p.to_dict()
    assert len(p.digest()) == 64
    for k in ("tokenizer", "bank_checksum", "feature_order", "class_map_hash", "partition_hash",
              "seed", "software_versions", "training_manifest_hash", "loss_mask_config"):
        assert k in d
    verify_provenance(d, p)                          # identical -> ok


def test_bank_checksum_or_feature_order_change_hard_fails_load():
    p = _prov()
    with pytest.raises(ValueError):
        verify_provenance(dict(p.to_dict(), bank_checksum="OTHER"), p)
    with pytest.raises(ValueError):
        verify_provenance(dict(p.to_dict(), feature_order=["shuffled"]), p)


def test_train_only_normalization_differs_from_full_cohort():
    # Safeguard 6: stats fit on TRAIN partition only must differ from a whole-cohort fit
    ft = fit_fusion(["a", "b"], [[0, 0], [2, 2]], "BC")
    ff = fit_fusion(["a", "b"], [[0, 0], [2, 2], [10, 10]], "BC")
    assert not (ft.mean == ff.mean).all()
