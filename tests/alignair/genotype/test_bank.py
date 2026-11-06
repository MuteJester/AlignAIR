"""A genotype is the chip's bank: the reference's subset of the alleles it names, refused when it names alleles the
reference does not hold or leaves a gene with none."""
import pytest

import GenAIRR.data as gd
from alignair.errors import InputError
from alignair.genotype.bank import genotype_bank
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component


@pytest.fixture(scope="module")
def igh():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def test_a_genotype_bank_keeps_only_the_named_alleles(igh):
    name = igh.gene("V").names[3]
    bank = genotype_bank(igh, {"v": [name]})
    assert bank.gene("V").names == [name]
    assert bank.gene("D").names == igh.gene("D").names and bank.gene("J").names == igh.gene("J").names


def test_a_genotype_naming_a_missing_allele_is_refused_with_the_hint(igh):
    with pytest.raises(InputError) as caught:
        genotype_bank(igh, {"V": ["IGHV9-99*01"]})
    assert "IGHV9-99*01" in str(caught.value) and "--reference" in caught.value.hint


def test_a_genotype_naming_a_gene_the_reference_lacks_is_refused():
    with pytest.raises(InputError, match="IGHD1-1\\*01"):
        genotype_bank(ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB), {"D": ["IGHD1-1*01"]})


def test_a_genotype_leaving_a_gene_empty_is_refused(igh):
    with pytest.raises(InputError, match="no allele"):
        genotype_bank(igh, {"V": []})


def test_a_genotype_bank_is_the_same_under_any_hash_seed():
    """A genotype read from a file is a set of names; the bank it gives must not follow the set's hash order, which
    changes from one process to the next."""
    import os
    import subprocess
    import sys
    code = ("from alignair.genotype.bank import genotype_bank; from alignair.reference.reference_set import "
            "ReferenceSet; names = [f'IGHV{i}-1*01' for i in range(30)]; "
            "ref = ReferenceSet.from_genotype({'V': {n: 'ACGT' * 3 for n in names}, 'J': {'IGHJ1*01': 'TTGG'}}); "
            "print(','.join(genotype_bank(ref, {'v': set(names[::2])}).gene('V').names))")
    banks = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": str(seed)},
                            capture_output=True, text=True, check=True).stdout.strip() for seed in range(4)}
    assert banks == {",".join(f"IGHV{i}-1*01" for i in range(0, 30, 2))}
