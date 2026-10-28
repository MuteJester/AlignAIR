"""Real simulated reads pass through the shared protein adapter, model and task loss."""
from itertools import islice

import pytest
import torch

from alignair.core import prepare_model
from alignair.data.adapters import build_protein_allele_batch
from alignair.data.projection import translated_record_usable
from alignair.reference import ReferenceSet
from alignair.tasks import build_tasks
from alignair.train.gym import Curriculum, build_experiment

pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_real_gym_protein_allele_batch_backpropagates_through_shared_model_and_tasks():
    import GenAIRR as ga
    source = ga.HUMAN_IGH_OGRDB
    setup = prepare_model(ReferenceSet.from_dataconfigs(source), input_space='protein', max_seq_length=192)
    assert setup.label_counts == {'v':198, 'd':33, 'j':7}
    parameters = {**Curriculum().params(.6), 'invert_d_prob': .05}
    experiment = build_experiment(source, parameters, allow_curatable=True)
    records = list(islice((record for record in islice(experiment.stream_records(n=None, seed=0), 512)
                           if translated_record_usable(record)), 8))
    assert len(records) == 8, 'not enough usable protein records in the bounded source fixture'
    inputs, targets = build_protein_allele_batch(records, setup.reference, setup.config)
    model = setup.build_model()
    tasks = build_tasks(setup.config)
    outputs = model(inputs)
    assert 'orientation_logits' not in outputs
    assert outputs['v_allele'].shape == (8,198)
    assert targets['v_allele'].shape == (8,198)
    total, _ = tasks.loss(outputs, targets, tasks.make_weights())
    assert torch.isfinite(total)
    total.backward()
    assert model.branches['v'].cls_head.weight.grad is not None


@pytest.fixture(scope='module')
def native_benchmark_cases():
    from alignair_benchmark import default_igh_spec, stream_benchmark
    return list(stream_benchmark(default_igh_spec(n_per_stratum=10, seed=1)))


@pytest.mark.parametrize('stratum,minimum,maximum', [
    ('clean_full', 9, 10), ('amplicon_fr1', 0, 1),
    ('amplicon_fr2', 0, 1), ('amplicon_janchor', 8, 10),
])
def test_native_benchmark_translation_availability_matches_read_layout(
        native_benchmark_cases, stratum, minimum, maximum):
    """Junction-retaining short reads remain translatable; FR crops lose that frame anchor."""
    records = [case.record for case in native_benchmark_cases if case.stratum == stratum]
    assert len(records) == 10
    usable = sum(translated_record_usable(record) for record in records)
    assert minimum <= usable <= maximum
