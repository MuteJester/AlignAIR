"""Lab candidates register when their package is imported, each under a versioned name, and each passes the
architecture extension contract: it trains, reloads through a model file and resumes. Data workers never build a
model, so a candidate registered only in the training process trains with them."""
import re

import numpy as np
import pytest
import torch

from alignair import TrainingConfig, run_training
from alignair.core import ModelComponents, components, register_architecture
from alignair.core.components import CHIP, architecture_components
from alignair.errors import ConfigurationError
from alignair.model_file import read_training_state
from alignair_lab import architectures
from alignair_lab.recipe import BATCH_SIZE, POLICY, SEED, training_config

pytestmark = pytest.mark.integration

SMALL = {'embed_dim': 4, 'trunk_channels': 8}


@pytest.fixture
def scratch_registry(monkeypatch):
    """Registrations made by the test are forgotten after it."""
    monkeypatch.setattr(components, '_ARCHITECTURES', dict(components._ARCHITECTURES))


def test_importing_the_package_registers_each_candidate_under_a_versioned_name():
    names = architectures.registered()
    assert 'lab-null-v2' in names and len(set(names)) == len(names)
    for name in names:
        assert re.fullmatch(architectures.NAME, name), name
        assert isinstance(architecture_components(name), ModelComponents)


def test_the_null_candidate_is_the_chip_under_its_own_name():
    assert architecture_components('lab-null-v2') == architecture_components('chip-v1')


@pytest.mark.parametrize('name', ['lab-null', 'lab-null-V1', 'Lab-null-v1', 'lab null-v1', 'lab-null-v2 '])
def test_an_unversioned_name_is_refused(scratch_registry, name):
    with pytest.raises(ConfigurationError, match='versioned'):
        architectures.register(name, CHIP)


@pytest.mark.parametrize('name', architectures.registered())
def test_a_candidate_trains_reloads_and_resumes(name, tmp_path):
    request = TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', steps=2, batch_size=2, device='cpu', seed=31,
                                          val_every=1, model_options={'architecture': name, **SMALL})
    losses = []
    run = run_training(request, output_dir=tmp_path / 'first', on_iteration=lambda item: losses.append(item.result.total))
    assert len(losses) == 2 and np.isfinite(losses).all()
    predictor = run.best_aligner(device='cpu')
    assert predictor.model.cfg.architecture == name
    raw = predictor.predict_raw(['ACGTN', 'AC'], batch_size=1)
    cfg = predictor.model.cfg
    for gene in cfg.genes:
        assert raw[f"{gene}_allele"].shape == (2, cfg.allele_count(gene))
    steps = []
    resumed = run_training(request.with_(steps=3, resume_path=run.model_path), output_dir=tmp_path / 'resumed',
                           on_iteration=lambda item: steps.append(item.step))
    assert steps == [3] and read_training_state(resumed.model_path).step == 3


def test_data_workers_never_build_a_model(scratch_registry, tmp_path):
    """The candidate is registered in this process only: a spawned worker that built a model could not find it."""
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.data.training import prepare_training_data
    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.assembly import TrainingStream, parallel_training_data
    register_architecture('lab-training-process-only-v1', CHIP)
    trained = run_training(training_config(steps=2, architecture='lab-training-process-only-v1', model_options=SMALL,
                                           device='cpu', data_workers=2), output_dir=tmp_path)
    assert trained.steps == 2
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), architecture='lab-training-process-only-v1',
                          **SMALL)
    stream = TrainingStream((gd.HUMAN_IGH_OGRDB,), POLICY, SEED, prepare_training_data(setup, device='cpu'))
    with parallel_training_data(stream, batch_size=BATCH_SIZE, workers=2, device='cpu', assemble_in_worker=True) \
            as batches:
        records = tuple(next(batches.records) for _ in range(BATCH_SIZE))
        assert isinstance(batches.batch_adapter(records).inputs['tokenized_sequence'], torch.Tensor)
