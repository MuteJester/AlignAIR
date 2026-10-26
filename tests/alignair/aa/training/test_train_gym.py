import pytest

pytestmark = pytest.mark.integration


def test_setup_builds_full_allele_aa_model():
    from alignair.aa.train_gym import setup_aa
    model, ref, cfg, logvars = setup_aa("HUMAN_IGH_OGRDB", max_seq_length=192)
    assert cfg.vocab_size == 23 and cfg.use_orientation is False and cfg.has_d is True
    assert cfg.v_allele_count == 198 and cfg.d_allele_count == 33 and cfg.j_allele_count == 7
    assert model.orientation_head is None
    assert len(ref.gene("V")) == 198


@pytest.mark.slow
def test_train_aa_runs_and_loss_is_finite():
    import torch

    from alignair.aa.train_gym import setup_aa, train_aa
    torch.manual_seed(0)
    model, ref, cfg, logvars = setup_aa("HUMAN_IGH_OGRDB", max_seq_length=192)
    history = train_aa(model, ref, "HUMAN_IGH_OGRDB", cfg, logvars,
                       steps=30, batch_size=16, lr=3e-4, seed=0, device="cpu", log_every=1000)
    assert len(history) == 30
    assert all(h["total"] == h["total"] for h in history)          # no NaN
    # segmentation + classification + the meta terms are all present (same scheme as DNA)
    for key in ("segmentation", "classification", "mutation", "indel", "productive"):
        assert key in history[-1]["parts"]
    # loss moved off its starting point
    assert history[-1]["total"] < history[0]["total"]
