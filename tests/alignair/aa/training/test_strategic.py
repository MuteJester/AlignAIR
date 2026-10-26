import pytest

pytestmark = pytest.mark.integration


def test_strategic_setup_class_head_dims():
    from alignair.aa.strategic import strategic_setup
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192)   # has_d=True by default now
    assert cfg.has_d is True and set(model.branches.keys()) == {"v", "d", "j"}
    assert cfg.v_allele_count == meta["n_v_classes"] == 149
    assert cfg.j_allele_count == meta["n_j_classes"] == 7
    assert cfg.d_allele_count == meta["n_d_alleles"] == 33 and len(meta["d_index"]) == 33
    out = model({"tokenized_sequence": __import__("torch").randint(1, 23, (2, 192))})
    assert out["v_class_logits"].shape == (2, 149) and out["j_class_logits"].shape == (2, 7)
    assert out["d_class_logits"].shape == (2, 33)


@pytest.mark.slow
def test_strategic_train_runs_on_real_gym_with_class_targets():
    import torch
    from alignair.aa.strategic import strategic_setup, strategic_train
    torch.manual_seed(0)
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192)   # V/D/J
    hist = strategic_train(model, cfg, class_maps, d_index=meta["d_index"], steps=30, batch_size=16,
                           lr=3e-4, sub_prob=0.5, seed=1, device="cpu", log_every=1000)
    assert len(hist) == 30
    assert all(h["total"] == h["total"] for h in hist)          # no NaN (incl. masked D loss)
    assert hist[-1]["total"] < hist[0]["total"]                 # learning
