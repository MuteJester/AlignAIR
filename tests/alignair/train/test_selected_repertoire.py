"""A selected repertoire: three quarters of a selectable curriculum component's records come from GenAIRR's
``productive_only()`` experiment. The V-anchored and fragment amplicons stay unselected: selection keeps the J
landmark observed, so it would suppress their 3' end loss and their short reads with it."""
from itertools import islice
from statistics import median

import pytest

from alignair.core import prepare_model
from alignair.model_file import read_metadata
from alignair.reference.reference_set import ReferenceSet
from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records, training_records
from alignair.train.trainer import train

pytestmark = pytest.mark.integration

# The 1.0.0 recipe's curriculum mix (progresses 0.3/0.6/0.9, heavy SHM 0.25, short boost 1): 7 components.
RECIPE = CurriculumStreamPolicy(progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1)
COMPONENTS = 7


def test_selected_records_are_productive_and_the_unselected_amplicons_keep_their_short_reads():
    import GenAIRR.data as gd
    selected = []                                  # per provider call: whether it draws from productive_only()

    def tagged(dataconfig, parameters, seed):
        call = len(selected)
        selected.append(bool(parameters.get('productive_only')))
        for record in stream_records(dataconfig, parameters, seed):
            yield {**record, 'call': call}

    records = list(islice(training_records((gd.HUMAN_IGH_OGRDB,), policy=RECIPE, seed=0, record_source=tagged),
                          COMPONENTS * 32))
    assert len(selected) == 28 and RECIPE.selected_shares() == (.75, .75, .75, 0., .75, 0., .75)
    for component, share in enumerate(RECIPE.selected_shares()):
        rows = records[component::COMPONENTS]
        picked = [row for row in rows if selected[row['call']]]
        assert len(picked) == share * len(rows) and all(row['productive'] is True for row in picked)
    for amplicon in (3, 5):                        # V-anchored and fragment reads: unselected, still short
        assert median(len(row['sequence']) for row in records[amplicon::COMPONENTS]) < 200


@pytest.mark.parametrize('options,fraction', [({}, 0.75), ({'selected_fraction': 0.5}, 0.5)])
def test_a_run_draws_its_selected_fraction_and_records_its_recipe(tmp_path, options, fraction):
    import GenAIRR.data as gd
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    setup = prepare_model(reference, embed_dim=4, filters=4, block_out=8)
    seen = []

    def watched(dataconfig, parameters, seed):
        seen.append(bool(parameters.get('productive_only')))
        return stream_records(dataconfig, parameters, seed)

    path = tmp_path / 'model.alignair'
    train(setup.build_model(), reference, [gd.HUMAN_IGH_OGRDB], setup.config, steps=1, batch_size=2,
          model_setup=setup, save_path=str(path), log_every=0, record_source=watched, **options)
    policy = CurriculumStreamPolicy(selected_fraction=fraction)
    assert seen == [name.startswith('selected') for streams in policy.substreams_for(gd.HUMAN_IGH_OGRDB)
                    for name, _ in streams]
    train_args = read_metadata(path)['training']['train_args']
    assert train_args['recipe'] == 'v3' and train_args['selected_fraction'] == fraction
    assert train_args['selected_shares'] == list(policy.selected_shares())
