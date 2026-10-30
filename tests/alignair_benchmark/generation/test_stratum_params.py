"""A benchmark stratum's resolved curriculum parameters are pinned: same spec name, version and seed, same data."""
import pytest

from alignair_benchmark.generation.genairr.experiments import resolve_stratum_params
from alignair_benchmark.generation.recipes.igh import default_igh_spec

pytestmark = pytest.mark.component


def test_strata_without_their_own_indel_counts_keep_the_historical_uniform_range():
    strata = [stratum for stratum in default_igh_spec().strata if "indel_count" not in stratum.param_overrides]
    assert len(strata) == 10
    for stratum in strata:
        assert resolve_stratum_params(stratum)["indel_count"] == (0, int(round(5 * stratum.progress))), stratum.name
    moderate = next(stratum for stratum in strata if stratum.name == "moderate_full")
    assert resolve_stratum_params(moderate)["indel_count"] == (0, 2)
