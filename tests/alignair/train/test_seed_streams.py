"""A run's record streams are independent: no curriculum component repeats another's records a few positions apart.

GenAIRR draws record k of the stream of seed s from seed s + k, so components seeded s, s + 1, ... would repeat each
other's V/D/J draws one position apart; each component draws its own reserved block of seeds instead.
"""
from itertools import islice

import pytest

from alignair.train.gym.streams import CurriculumStreamPolicy, training_records

pytestmark = pytest.mark.integration

RECIPE = CurriculumStreamPolicy(progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1)   # 7 components
COMPONENTS = 7


def matching(left, right, shift):
    """The share of positions k at which ``left[k] == right[k + shift]``."""
    pairs = [(left[k], right[k + shift]) for k in range(len(left)) if 0 <= k + shift < len(right)]
    return sum(a == b for a, b in pairs) / len(pairs)


def test_the_recipes_seven_components_are_not_shifted_copies_of_each_other():
    import GenAIRR.data as gd
    records = list(islice(training_records((gd.HUMAN_IGH_OGRDB,), policy=RECIPE, seed=0), COMPONENTS * 80))
    calls = [[(record['v_call'], record['d_call'], record['j_call']) for record in records[component::COMPONENTS]]
             for component in range(COMPONENTS)]
    shares = {(a, b, shift): matching(calls[a], calls[b], shift)
              for a in range(COMPONENTS) for b in range(COMPONENTS) if a != b for shift in range(-8, 9)}
    assert max(shares.values()) < 0.1, max(shares.items(), key=lambda item: item[1])
