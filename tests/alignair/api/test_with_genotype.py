"""``Aligner.with_genotype``: an Aligner whose calls come from a donor's genotype, the genotype's subset of the
reference bound to a copy of the model once. Its records are the ones the model bound to that bank gives (the
mechanism the removed ``predict(genotype=)`` rebound the shared model with on every call)."""
import copy
import json

import pytest

from alignair import Aligner
from alignair.genotype.bank import genotype_bank
from alignair.errors import UnknownAlleleError
from tests.alignair.predict.golden_rows import GOLDEN_READS, TEST_CHIP

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def aligner():
    return Aligner.from_pretrained(str(TEST_CHIP), device="cpu")


@pytest.fixture(scope="module")
def reads():
    return json.loads(GOLDEN_READS.read_text())["reads"]


def _genotype(reference):
    return {"v": list(reference.gene("V").names[:6]), "j": list(reference.gene("J").names[:3])}


def _rows(result):
    return json.loads(json.dumps(result.records, default=str, sort_keys=True))


def test_with_genotype_gives_the_records_of_the_model_bound_to_the_genotypes_bank(aligner, reads):
    genotype = _genotype(aligner.reference)
    bank = genotype_bank(aligner.reference, genotype)
    model = copy.deepcopy(aligner.model)
    model.bind_reference(bank)
    expected = Aligner(model, bank, device="cpu").predict(reads, batch_size=8)
    assert _rows(aligner.with_genotype(genotype).predict(reads, batch_size=8)) == _rows(expected)


def test_with_genotype_calls_only_the_genotypes_alleles_and_leaves_this_aligner_whole(aligner, reads):
    before = _rows(aligner.predict(reads, batch_size=8))
    genotype = _genotype(aligner.reference)
    donor = aligner.with_genotype(genotype)
    assert donor.reference.gene("V").names == genotype["v"]
    assert donor.reference.gene("D").names == aligner.reference.gene("D").names     # a gene it does not name: all
    for record in donor.predict(reads, batch_size=8):
        assert set(filter(None, record["v_call"].split(","))) <= set(genotype["v"])
    assert _rows(aligner.predict(reads, batch_size=8)) == before


def test_with_genotype_refuses_an_allele_the_reference_lacks(aligner):
    with pytest.raises(UnknownAlleleError, match="IGHV-missing"):
        aligner.with_genotype({"V": ["IGHV-missing*01"]})


@pytest.mark.parametrize("genotype", [{"V": "IGHV1-2*02"}, {"v": None}, {"J": 3}])
def test_with_genotype_refuses_a_segment_given_one_string_or_no_list_as_a_genotype_file_is(aligner, genotype):
    """A segment's entry is a list of allele names; one string is not split into characters (as the CLI's genotype file
    refuses it)."""
    from alignair.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match="entry is .*not a list of allele names") as caught:
        aligner.with_genotype(genotype)
    assert not isinstance(caught.value, UnknownAlleleError) and caught.value.hint
