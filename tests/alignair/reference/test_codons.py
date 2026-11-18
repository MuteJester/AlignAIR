"""One definition each for the codons and IMGT columns the anchors, the novel alleles and the AIRR fields read."""
import pytest

from alignair.predict.airr.constants import IMGT_REGIONS
from alignair.reference.codons import (CONSERVED_CODONS, CYS_CODONS, GAP, IMGT_CODON_104, J_ANCHOR_CODONS,
                                       STOP_CODONS)
from alignair.reference.anchors import imgt_position

pytestmark = pytest.mark.component


def test_the_codons_and_the_gap():
    assert STOP_CODONS == {"TAA", "TAG", "TGA"} and GAP == "."
    assert CONSERVED_CODONS == {"V": CYS_CODONS, "J": J_ANCHOR_CODONS}
    assert set(CYS_CODONS) == {"TGT", "TGC"} and set(J_ANCHOR_CODONS) == {"TGG", "TTT", "TTC"}


def test_codon_104_starts_at_the_column_fwr3_ends_before():
    assert IMGT_CODON_104 == 309 and IMGT_REGIONS["fwr3"][1] == IMGT_CODON_104 + 3
    assert tuple(IMGT_REGIONS) == ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3")
    gapped = "A" * IMGT_CODON_104 + "TGG" + "AAA"
    assert imgt_position(gapped, gapped) == (IMGT_CODON_104, "TGG")
