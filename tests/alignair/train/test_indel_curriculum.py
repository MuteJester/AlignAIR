"""The training stream carries realistic polymerase indels: most reads none, and a run records the indel range."""
from collections import Counter
from itertools import islice

import pytest

from alignair.core import prepare_model
from alignair.model_file import read_metadata
from alignair.reference.reference_set import ReferenceSet
from alignair.train.records.curriculum import Curriculum
from alignair.train.records.streams import CurriculumStreamPolicy, stream_records, training_records
from alignair.train.config import TrainingConfig
from alignair.train.trainer import _train

pytestmark = pytest.mark.integration

# The 1.0.0 recipe's curriculum mix (progresses 0.3/0.6/0.9, heavy SHM 0.25, short boost 1): 7 components.
RECIPE = CurriculumStreamPolicy(progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1)


@pytest.mark.slow                                 # 2,100 records, most of them selected: about 45 s
def test_the_recipe_stream_is_about_85_percent_indel_free_and_deterministic_per_seed():
    import GenAIRR.data as gd
    records = list(islice(training_records((gd.HUMAN_IGH_OGRDB,), policy=RECIPE, seed=0), 2100))
    counts = Counter(int(record['n_indels']) for record in records)
    assert counts[0] / len(records) == pytest.approx(0.85, abs=0.03)
    assert set(counts) <= {0, 1, 2, 3} and counts[1] > counts[2] > counts[3]
    again = list(islice(training_records((gd.HUMAN_IGH_OGRDB,), policy=RECIPE, seed=0), 140))
    assert again == records[:140]


@pytest.mark.parametrize('options,form', [({}, 'weighted'), ({'indel_form': 'uniform'}, 'uniform')])
def test_a_run_draws_its_components_from_its_indel_form_and_fraction_and_records_them(tmp_path, options, form):
    import GenAIRR.data as gd
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    setup = prepare_model(reference, embed_dim=4, trunk_channels=8)
    seen = []

    def watched(dataconfig, parameters, seed):
        seen.append(parameters['indel_count'])
        return stream_records(dataconfig, parameters, seed)

    path = tmp_path / 'model.alignair'
    _train(setup, setup.build_model(), [gd.HUMAN_IGH_OGRDB],
           TrainingConfig(dataconfigs=(gd.HUMAN_IGH_OGRDB,), steps=1, batch_size=2),
           CurriculumStreamPolicy(indel_fraction=(0.1, 0.2), **options), save_path=str(path), log_every=0,
           record_source=watched)
    curriculum = Curriculum(indel_form=form, indel_fraction=(0.1, 0.2))
    period = len(CurriculumStreamPolicy().selection_pattern)          # the streams of one component
    assert seen[::period][:3] == [curriculum.indel_count(progress) for progress in (0.3, 0.6, 0.9)]
    train_args = read_metadata(path)['training']['train_args']
    assert train_args['indel_form'] == form and train_args['indel_fraction'] == [0.1, 0.2]
