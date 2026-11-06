"""A GenAIRR genotype fixes which alleles recombine, so it cannot be combined with allele restrictions or weights."""
import pytest

import GenAIRR.data as gd
from alignair.errors import ConfigurationError
from alignair.train.gym.experiment import build_experiment

pytestmark = pytest.mark.component


@pytest.mark.parametrize("other", [{"restrict_alleles": {"V": ["IGHV1-2*02"]}},
                                   {"v_allele_weights": {"IGHV1-2*02": 1.0}}], ids=["restrict_alleles", "weights"])
@pytest.mark.parametrize("genotype", [{"genotype_seed": 3}], ids=["sampled"])
def test_a_genotype_with_restricted_alleles_or_weights_is_refused(genotype, other):
    with pytest.raises(ConfigurationError) as caught:
        build_experiment(gd.HUMAN_IGH_OGRDB, {**genotype, **other})
    message = str(caught.value)
    assert "genotype" in message and next(iter(other)) in message
