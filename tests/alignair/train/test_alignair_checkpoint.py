import pytest
import GenAIRR.data as gd
import torch
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair.tasks.losses import make_loss_weights
from alignair.train.checkpoints import save_checkpoint, load_training_checkpoint
from alignair import model_file as mf

pytestmark = pytest.mark.integration


def test_trainer_save_checkpoint_writes_alignair(tmp_path):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    model = AlignAIR(cfg)
    logvars = make_loss_weights(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    p = tmp_path / "m.alignair"
    save_checkpoint(str(p), cfg, model, logvars, step=1000, opt=opt,
                    dataconfigs=[gd.HUMAN_IGH_OGRDB], train_args={"lr": 1e-4, "batch_size": 64})
    assert mf.container.is_alignair_file(str(p))
    md = mf.read_metadata(str(p))
    assert md["training"]["train_args"]["batch_size"] == 64
    ts = load_training_checkpoint(p)
    assert ts.step == 1000 and ts.optimizer_state is not None
    assert "python" in ts.rng and "torch" in ts.rng


def test_a_historical_productive_objective_is_refused_before_any_state_is_restored(tmp_path):
    from dataclasses import replace
    import random
    from alignair.core import prepare_model
    from alignair.errors import ConfigurationError
    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.trainer import train
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    setup = prepare_model(reference, embed_dim=4, filters=4, block_out=8)
    historical = replace(setup.config, productive_unknown='negative')
    old_model, old_weights = AlignAIR(historical), make_loss_weights(historical)
    old_optimizer = torch.optim.AdamW([*old_model.parameters(), *old_weights.parameters()], lr=1e-4)
    path = tmp_path / 'historical.alignair'
    save_checkpoint(str(path), historical, old_model, old_weights, step=2, opt=old_optimizer,
                    dataconfigs=[gd.HUMAN_IGH_OGRDB], label_schema=setup.label_schema())
    model = setup.build_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    torch.manual_seed(123)
    random.seed(123)
    rng = torch.get_rng_state(), random.getstate()
    with pytest.raises(ConfigurationError, match='productive-label contract') as caught:
        train(model, reference, [gd.HUMAN_IGH_OGRDB], setup.config, steps=4, batch_size=2, model_setup=setup,
              resume_path=str(path), log_every=0)
    assert caught.value.context['field'] == 'productive_unknown'
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng[0]) and random.getstate() == rng[1]
