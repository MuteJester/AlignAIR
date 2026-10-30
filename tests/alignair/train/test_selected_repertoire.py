"""A selected repertoire: three quarters of a selectable curriculum component's records come from GenAIRR's
``productive_only()`` experiment. The V-anchored and fragment amplicons stay unselected: selection keeps the J
landmark observed, so it would suppress their 3' end loss and their short reads with it."""
from itertools import islice
from statistics import median

import pytest

from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records, training_records

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
