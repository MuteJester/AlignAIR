import torch
from alignair.aa.resume import load_checkpoint
from alignair.aa.strategic import strategic_setup
from alignair.aa.strategic_trunc import stage1_train


def test_resume_restores_optimizer_and_advances_step(tmp_path):
    model, cfg, cmaps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    d = str(tmp_path)
    # 6 steps, checkpoint at 6
    stage1_train(model, cfg, cmaps, meta, steps=6, batch_size=8, base_lr=1e-4, lr_min=1e-5,
                 total_budget=12, seed=0, device="cpu", log_every=0, ckpt_dir=d, ckpt_steps=(6,))
    ck = load_checkpoint(f"{d}/model_cont_step6.pt")
    assert ck["step"] == 6 and ck["optimizer"] is not None and ck["optimizer"]["state"]
    # resume for 6 more; step advances to 12, optimizer state still present
    model2, cfg2, cmaps2, meta2 = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    stage1_train(model2, cfg2, cmaps2, meta2, steps=6, batch_size=8, base_lr=1e-4, lr_min=1e-5,
                 total_budget=12, seed=0, device="cpu", log_every=0, ckpt_dir=d, ckpt_steps=(12,),
                 resume_path=f"{d}/model_cont_step6.pt")
    ck2 = load_checkpoint(f"{d}/model_cont_step12.pt")
    assert ck2["step"] == 12
