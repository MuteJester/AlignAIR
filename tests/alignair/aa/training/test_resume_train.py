import pytest
import torch
from alignair.aa.resume import load_checkpoint, save_checkpoint
from tests.support.model_setup import protein_class_setup
from alignair.aa import strategic_trunc as st
from alignair.aa.strategic_trunc import stage1_train

pytestmark = pytest.mark.integration


def test_resume_restores_optimizer_and_advances_step(tmp_path):
    setup = protein_class_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    model = setup.build_model()
    cfg, cmaps = setup.config, setup.protein_reference.class_maps
    meta = {'d_index': dict(setup.reference.gene('D').index) if setup.config.has_d else {}}
    d = str(tmp_path)
    # 6 steps, checkpoint at 6
    stage1_train(model, cfg, cmaps, meta, steps=6, batch_size=8, base_lr=1e-4, lr_min=1e-5,
                 total_budget=12, seed=0, device="cpu", log_every=0, ckpt_dir=d, ckpt_steps=(6,))
    ck = load_checkpoint(f"{d}/model_cont_step6.pt")
    assert ck["step"] == 6 and ck["optimizer"] is not None and ck["optimizer"]["state"]
    # resume for 6 more; step advances to 12, optimizer state still present
    setup = protein_class_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    model2 = setup.build_model()
    cfg2, cmaps2 = setup.config, setup.protein_reference.class_maps
    meta2 = {'d_index': dict(setup.reference.gene('D').index) if setup.config.has_d else {}}
    stage1_train(model2, cfg2, cmaps2, meta2, steps=6, batch_size=8, base_lr=1e-4, lr_min=1e-5,
                 total_budget=12, seed=0, device="cpu", log_every=0, ckpt_dir=d, ckpt_steps=(12,),
                 resume_path=f"{d}/model_cont_step6.pt")
    ck2 = load_checkpoint(f"{d}/model_cont_step12.pt")
    assert ck2["step"] == 12


def test_resume_reseeds_gym_corpus_stream_from_seed_plus_start_step(tmp_path, monkeypatch):
    """Spec §2: "the augmentation RNG + gym stream re-seed deterministically from (seed, step)".
    The local augmentation Generator already does this (`np.random.default_rng([seed, start_step])`)
    but the GYM/CORPUS record stream must ALSO incorporate start_step (mirroring
    `trainer.py`'s `_build_stream(seed + start)`) — otherwise every resume replays the corpus from
    record 0 instead of walking a fresh slice. Captures the seed actually passed to `training_records`
    (monkeypatched to avoid touching the real GenAIRR pipeline) instead of relying on end-to-end
    behavior, since that is deterministic and fast."""
    captured = []

    def fake_training_records(dataconfigs, *, policy, seed, translated_only):
        assert policy.mode == "amplicon" and translated_only
        captured.append(seed)
        return iter(())                       # empty -> batch loop breaks after 0 steps

    monkeypatch.setattr(st, "training_records", fake_training_records)

    # -- fresh run: start_step == 0, stream seed must be seed + 0 --
    setup = protein_class_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    model = setup.build_model()
    cfg, cmaps = setup.config, setup.protein_reference.class_maps
    meta = {'d_index': dict(setup.reference.gene('D').index) if setup.config.has_d else {}}
    stage1_train(model, cfg, cmaps, meta, steps=1, batch_size=8, seed=7, device="cpu", log_every=0)
    assert captured[-1] == 7

    # -- resumed run: build a full-state checkpoint stopped at step=50, then resume from it --
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
    ck_path = str(tmp_path / "model_cont_step50.pt")
    save_checkpoint(ck_path, model=model, cfg={}, step=50, optimizer=opt, scheduler=sch,
                    lr=1e-4, history=[], provenance={"sample_policy": st.SAMPLE_POLICY_ID})

    stage1_train(model, cfg, cmaps, meta, steps=1, batch_size=8, seed=7, device="cpu", log_every=0,
                 resume_path=ck_path)
    assert captured[-1] == 7 + 50               # NOT 7 — the resumed run must walk a fresh slice


def test_checkpoint_provenance_carries_created_step(tmp_path):
    """Spec §2 provenance schema: {git_sha, base_model_sha256, seed, hyperparams, dc_name,
    created_step}. `created_step` was missing (only the top-level `step` field carried it)."""
    setup = protein_class_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    model = setup.build_model()
    cfg, cmaps = setup.config, setup.protein_reference.class_maps
    meta = {'d_index': dict(setup.reference.gene('D').index) if setup.config.has_d else {}}
    d = str(tmp_path)
    stage1_train(model, cfg, cmaps, meta, steps=6, batch_size=8, base_lr=1e-4, lr_min=1e-5,
                 total_budget=6, seed=0, device="cpu", log_every=0, ckpt_dir=d, ckpt_steps=(6,))
    ck = load_checkpoint(f"{d}/model_cont_step6.pt")
    assert ck["provenance"]["created_step"] == 6 == ck["step"]


@pytest.mark.parametrize('policy',[None,'protein-crop-read-offset-v0'])
def test_incompatible_resume_is_rejected_before_model_mutation(tmp_path,policy):
    from dataclasses import replace
    from alignair.core import AlignAIR
    from alignair.errors import ConfigurationError
    setup = protein_class_setup(max_seq_length=192,has_d=False)
    cfg, maps = setup.config, setup.protein_reference.class_maps
    meta = {'d_index': dict(setup.reference.gene('D').index) if setup.config.has_d else {}}
    cfg=replace(cfg,embed_dim=4,filters=4,block_out=8)
    model=AlignAIR(cfg)
    state={key:value.clone() for key,value in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10)
    path=tmp_path/'old.pt'
    save_checkpoint(path,model=model,cfg={},step=3,optimizer=opt,scheduler=sch,
                    lr=1e-4,history=[],provenance={'sample_policy':policy})
    with pytest.raises(ConfigurationError,match='policy differs') as exc:
        stage1_train(model,cfg,maps,meta,steps=1,resume_path=path)
    assert 'init_path' in exc.value.hint
    for key,value in model.state_dict().items(): torch.testing.assert_close(value,state[key])
    # Explicit warm starts reuse weights with a fresh optimizer and a fresh policy.
    history=stage1_train(model,cfg,maps,meta,steps=1,batch_size=2,init_path=path,log_every=0,p_crop=1.)
    assert history[0]['step']==1 and set(history[0]['class_support'])=={'v','j'}


def test_policy_is_stamped_on_new_continuation_checkpoints(tmp_path):
    from dataclasses import replace
    from alignair.core import AlignAIR
    setup = protein_class_setup(max_seq_length=192,has_d=False)
    cfg, maps = setup.config, setup.protein_reference.class_maps
    meta = {'d_index': dict(setup.reference.gene('D').index) if setup.config.has_d else {}}
    cfg=replace(cfg,embed_dim=4,filters=4,block_out=8)
    model=AlignAIR(cfg)
    history=stage1_train(model,cfg,maps,meta,steps=1,batch_size=2,ckpt_dir=tmp_path,
                         ckpt_steps=(1,),log_every=0,p_crop=1.)
    ck=load_checkpoint(tmp_path/'model_cont_step1.pt')
    assert ck['provenance']['sample_policy']==st.SAMPLE_POLICY_ID
    assert all(sum(reasons.values())==2 for reasons in history[0]['class_reasons'].values())
