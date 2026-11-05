"""A selected repertoire: three quarters of a full-length curriculum component's records come from GenAIRR's
``productive_only()`` experiment. Every amplicon component stays unselected: selection keeps the V and J landmarks
observed, so it would drop the truncated reads that miss an anchor, which the amplicons exist for. A reference GenAIRR
cannot draw productive records from trains unselected, and the run records it."""
from itertools import islice
import os

import pytest

from alignair.core import prepare_model
from alignair.model_file import read_metadata
from alignair.reference.reference_set import ReferenceSet
from alignair.train.gym.streams import (CurriculumStreamPolicy, dataconfig_name, selection_supported, stream_records,
                                       training_records)
from alignair.train.trainer import train

pytestmark = pytest.mark.integration

# An incompletely annotated custom reference: GenAIRR finds no stop-free coding V allele in it.
CUSTOM = {'v_fasta': 'examples/custom_reference/v.fasta', 'j_fasta': 'examples/custom_reference/j.fasta',
          'd_fasta': 'examples/custom_reference/d.fasta'}

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
                          COMPONENTS * 64))
    assert len(selected) == 28 and RECIPE.selected_shares() == (.75, .75, .75, 0., 0., 0., .75)
    for component, share in enumerate(RECIPE.selected_shares()):
        rows = records[component::COMPONENTS]
        picked = [row for row in rows if selected[row['call']]]
        assert len(picked) == share * len(rows) and all(row['productive'] is True for row in picked)
    # The amplicons (V-anchored, J-anchored, fragment) keep the reads selection drops. GenAIRR's productive_only()
    # draws no read of unknown productivity (one that misses an anchor) and no J-anchored read under 60 nt. Unselected,
    # the amplicons draw about half their reads of unknown productivity, the J-anchored one a quarter, and a fifth of
    # its reads under 60 nt; drawn from three selected streams of four, as the amplicons were before, they would hold
    # a quarter of that. Here the J-anchored reads hold 18 and 12 of 64, where the selected design drew 5 and 5.
    amplicons = [row for amplicon in (3, 4, 5) for row in records[amplicon::COMPONENTS]]
    assert sum(row['productive'] is None for row in amplicons) >= len(amplicons) / 4
    anchored = records[4::COMPONENTS]
    assert sum(row['productive'] is None for row in anchored) >= len(anchored) / 6
    assert sum(len(row['sequence']) < 60 for row in anchored) >= len(anchored) / 8


def watched_run(tmp_path, dataconfig, **options):
    """Train one step on ``dataconfig`` with ``train`` ``options``; return whether each stream drew under
    productive_only(), and the run's train_args."""
    reference = ReferenceSet.from_dataconfigs(dataconfig)
    setup = prepare_model(reference, embed_dim=4, trunk_channels=8)
    seen = []

    def watched(dataconfig, parameters, seed):
        seen.append(bool(parameters.get('productive_only')))
        return stream_records(dataconfig, parameters, seed)

    path = tmp_path / 'model.alignair'
    train(setup.build_model(), reference, [dataconfig], setup.config, steps=1, batch_size=2, model_setup=setup,
          save_path=str(path), log_every=0, record_source=watched, **options)
    return seen, read_metadata(path)['training']['train_args']


@pytest.mark.parametrize('options,fraction', [({}, 0.75), ({'selected_fraction': 0.5}, 0.5)])
def test_a_run_draws_its_selected_fraction_and_records_its_recipe(tmp_path, options, fraction):
    import GenAIRR.data as gd
    seen, train_args = watched_run(tmp_path, gd.HUMAN_IGH_OGRDB, **options)
    policy = CurriculumStreamPolicy(selected_fraction=fraction)
    assert seen == [name.startswith('selected') for streams in policy.substreams_for(gd.HUMAN_IGH_OGRDB)
                    for name, _ in streams]
    assert train_args['recipe'] == 'v3' and train_args['selected_fraction'] == fraction
    assert train_args['selected_shares'] == list(policy.selected_shares())
    assert train_args['selection_unavailable'] == []


@pytest.mark.skipif(not all(os.path.exists(path) for path in CUSTOM.values()), reason='example FASTAs not present')
def test_a_reference_genairr_cannot_select_from_trains_unselected_and_the_run_says_so(tmp_path, capsys):
    import GenAIRR.data as gd
    from alignair.train.build import build_dataconfigs
    (custom,), _ = build_dataconfigs(chain_type='BCR_HEAVY', **CUSTOM)
    assert selection_supported(gd.HUMAN_IGH_OGRDB) and not selection_supported(custom)
    seen, train_args = watched_run(tmp_path, custom)
    assert len(seen) == COMPONENTS and not any(seen)
    assert train_args['selected_fraction'] == 0.75 and train_args['selected_shares'] == [0.] * COMPONENTS
    assert train_args['selection_unavailable'] == [dataconfig_name(custom)]
    assert 'selection unavailable' in capsys.readouterr().out
