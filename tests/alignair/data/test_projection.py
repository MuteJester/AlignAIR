import pytest
from alignair.data.projection import CODON_TABLE, translate_nt

pytestmark = pytest.mark.component


def test_codon_table_has_64_entries_and_known_codons():
    assert len(CODON_TABLE) == 64
    assert CODON_TABLE["ATG"] == "M"
    assert CODON_TABLE["TGG"] == "W"
    assert CODON_TABLE["TGT"] == "C" and CODON_TABLE["TGC"] == "C"
    assert CODON_TABLE["TAA"] == CODON_TABLE["TAG"] == CODON_TABLE["TGA"] == "*"


def test_translate_nt_basic_frame_unknown_and_partial():
    assert translate_nt("ATGTGG") == "MW"
    assert translate_nt("TGGTAG") == "W*"
    assert translate_nt("AATGTGG", frame=1) == "MW"      # skip 1 base, then ATG TGG
    assert translate_nt("ATGA") == "M"                    # trailing partial dropped
    assert translate_nt("ATGNNN") == "MX"                 # non-ACGT codon -> X
    assert translate_nt("atgtgg") == "MW"                 # lowercase ok
    assert translate_nt('AUGUGG') == 'MW'                  # RNA uses the same code
    assert translate_nt('ATGNNNTAG', unknown='?', stop='!') == 'M?!'
