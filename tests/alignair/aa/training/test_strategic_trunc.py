import pytest

pytestmark = pytest.mark.integration


def test_prototypes_by_class_covers_all_v_classes():
    import GenAIRR as ga
    from alignair.reference.protein import build_class_maps
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


def test_validation_callback_cannot_leave_following_steps_in_evaluation_mode():
    from dataclasses import replace
    from alignair.core import AlignAIR
    from alignair.aa.strategic import strategic_setup
    from alignair.aa.strategic_trunc import stage1_train
    _, cfg, maps, meta = strategic_setup(max_seq_length=192, has_d=True)
    cfg = replace(cfg, embed_dim=4, filters=4, block_out=8)
    model = AlignAIR(cfg)
    training_modes, callback_modes = [], []
    handle = model.register_forward_pre_hook(lambda module, _: training_modes.append(module.training))
    def validate(candidate, step):
        callback_modes.append(candidate.training)
        candidate.eval()
        return {'objective': float(step)}
    try:
        history = stage1_train(model, cfg, maps, meta, steps=2, batch_size=2, log_every=0,
                               val_every=1, val_fn=validate, p_crop=0., seed=0)
    finally:
        handle.remove()
    assert len(history) == 2
    assert training_modes == [True, True] and callback_modes == [False, False]
    assert model.training
