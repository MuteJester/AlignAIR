"""The training reference as GenAIRR dataconfigs: built-in names, or a custom reference from V, D and J FASTA files;
a request that cannot build one is a ConfigurationError naming the field."""
import os

import pytest

from alignair.errors import ConfigurationError
from alignair.genairr.custom import build_dataconfigs

pytestmark = pytest.mark.integration

_V = "examples/custom_reference/v.fasta"
_D = "examples/custom_reference/d.fasta"
_J = "examples/custom_reference/j.fasta"
_have_fasta = all(os.path.exists(p) for p in (_V, _D, _J))


def test_built_in_names_give_their_dataconfigs_and_no_build_report():
    dcs, report = build_dataconfigs(dataconfig=["HUMAN_IGH_OGRDB"])
    assert len(dcs) == 1 and report is None and dcs[0].metadata.has_d


@pytest.mark.parametrize("request_, message, field", [
    ({"dataconfig": ["NOT_A_REAL_CONFIG"]}, "unknown GenAIRR dataconfig", "dataconfigs"),
    ({"v_fasta": _V}, "needs a V FASTA, a J FASTA and a chain type", "j_fasta"),
    ({"dataconfig": ["HUMAN_IGH_OGRDB"], "v_fasta": "v.fa", "j_fasta": "j.fa", "chain_type": "BCR_HEAVY"},
     "choose one reference", "dataconfig"),
    ({"v_fasta": _V, "j_fasta": _J, "chain_type": "BCR_HEAVY"}, "carries a D gene", "d_fasta"),
    ({"v_fasta": _V, "j_fasta": _J, "d_fasta": _D, "chain_type": "BCR_LIGHT_KAPPA"}, "no D gene", "d_fasta"),
    ({"v_fasta": "no/such/v.fasta", "j_fasta": _J, "chain_type": "BCR_LIGHT_KAPPA"}, "does not exist", "v_fasta"),
])
def test_a_reference_that_cannot_be_built_names_its_field(request_, message, field):
    with pytest.raises(ConfigurationError, match=message) as caught:
        build_dataconfigs(**request_)
    assert caught.value.context.get("field") == field


@pytest.mark.skipif(not _have_fasta, reason="custom_reference example FASTAs not present")
def test_a_custom_reference_is_built_from_its_fasta_files():
    dcs, report = build_dataconfigs(v_fasta=_V, j_fasta=_J, d_fasta=_D, chain_type="BCR_HEAVY")
    assert len(dcs) == 1 and dcs[0].metadata.has_d
    assert report is not None                          # cartridge build report (warnings/rejected)
