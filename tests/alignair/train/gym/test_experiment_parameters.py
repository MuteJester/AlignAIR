"""``build_experiment`` simulates with the parameters the curriculum, the record streams and the benchmark give it: a
key it does not read is refused (a typo would otherwise simulate the default silently), and the share of inverted Ds
is always given, never a hidden default."""
import pytest

import GenAIRR.data as gd
from alignair.errors import ConfigurationError
from alignair.train.gym import Curriculum
from alignair.train.gym.experiment import build_experiment

pytestmark = pytest.mark.component


def _params(**extra):
    return {**Curriculum().params(0.3), "invert_d_prob": 0.0, **extra}


@pytest.mark.parametrize("key", ["genotype_seed", "receptor_revision_prob", "paired_end", "mutation_rat"])
def test_a_key_the_builder_does_not_read_is_refused(key):
    with pytest.raises(ConfigurationError, match=key) as caught:
        build_experiment(gd.HUMAN_IGH_OGRDB, _params(**{key: 1}))
    assert caught.value.context["field"] == key


def test_the_share_of_inverted_ds_is_required():
    params = _params()
    del params["invert_d_prob"]
    with pytest.raises(ConfigurationError, match="invert_d_prob"):
        build_experiment(gd.HUMAN_IGH_OGRDB, params)
