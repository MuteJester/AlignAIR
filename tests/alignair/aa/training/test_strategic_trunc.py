import pytest

pytestmark = pytest.mark.integration


def test_prototypes_by_class_covers_all_v_classes():
    import GenAIRR as ga
    from alignair.aa.data import build_class_maps
    from alignair.aa.strategic_trunc import prototypes_by_class
    cm, _ = build_class_maps("HUMAN_IGH_OGRDB")
    p = prototypes_by_class(ga.HUMAN_IGH_OGRDB, cm["V"], "V")
    assert len(p) == 149 and all(isinstance(v, str) and v for v in p.values())


@pytest.mark.slow
def test_stage1_train_runs_with_crops_and_partial_labels():
    import torch
    from alignair.aa.strategic import strategic_setup
    from alignair.aa.strategic_trunc import stage1_train
    torch.manual_seed(0)
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    hist = stage1_train(model, cfg, class_maps, meta, steps=40, batch_size=16, lr=3e-4,
                        p_crop=0.4, p_sub=0.4, p_il=0.15, seed=1, device="cpu", log_every=1000)
    assert len(hist) == 40
    assert all(h["total"] == h["total"] for h in hist)          # no NaN (partial-label + masked genes)
    assert hist[-1]["total"] < hist[0]["total"]
