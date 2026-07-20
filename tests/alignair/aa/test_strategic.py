import pytest


def test_strategic_setup_class_head_dims():
    from alignair.aa.strategic import strategic_setup
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192)
    assert cfg.has_d is False                       # V/J only; AAAlignAIR has no orientation head at all
    assert model.branches.keys() == {"v", "j"}
    assert cfg.v_allele_count == meta["n_v_classes"] == 149
    assert cfg.j_allele_count == meta["n_j_classes"] == 7
    out = model({"tokenized_sequence": __import__("torch").randint(1, 23, (2, 192))})
    assert out["v_class_logits"].shape == (2, 149) and out["j_class_logits"].shape == (2, 7)


@pytest.mark.slow
def test_strategic_train_runs_on_real_gym_with_class_targets():
    import torch
    from alignair.aa.strategic import strategic_setup, strategic_train
    torch.manual_seed(0)
    model, cfg, class_maps, _ = strategic_setup("HUMAN_IGH_OGRDB", 192)
    hist = strategic_train(model, cfg, class_maps, steps=30, batch_size=16, lr=3e-4,
                           sub_prob=0.5, seed=1, device="cpu", log_every=1000)
    assert len(hist) == 30
    assert all(h["total"] == h["total"] for h in hist)          # no NaN
    assert hist[-1]["total"] < hist[0]["total"]                 # learning
