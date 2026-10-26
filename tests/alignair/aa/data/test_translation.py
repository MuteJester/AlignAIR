import pytest
from alignair.aa.translation import (
    CODON_TABLE, translate_nt, frame_offset_of, project_codons, AAProjection,
)

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


def test_frame_offset_of():
    assert frame_offset_of(285) == 0
    assert frame_offset_of(286) == 1
    assert frame_offset_of(287) == 2
    assert frame_offset_of(None) is None


def test_project_codons_clean_vj_no_boundary_cut():
    # frame 0, V=[0,6) J=[6,12): residues r0,r1 in V ; r2,r3 in J ; no straddle.
    proj = project_codons(12, 0, {"V": (0, 6), "J": (6, 12)})
    assert proj.n_residues == 4
    assert proj.provenance == ("V", "V", "J", "J")
    assert proj.aa_bounds["V"] == (0, 2) and proj.aa_bounds["J"] == (2, 4)
    assert proj.phase["V"] == {"start": 0, "end": 0}
    assert proj.phase["J"] == {"start": 0, "end": 0}


def test_project_codons_boundary_straddle_is_masked():
    # frame 0, V=[0,5) J=[5,12): residue r1 spans nt[3,6) crossing the 5-boundary -> None (masked).
    proj = project_codons(12, 0, {"V": (0, 5), "J": (5, 12)})
    assert proj.provenance == ("V", None, "J", "J")
    assert proj.aa_bounds["V"] == (0, 1)          # only r0 fully owned by V
    assert proj.aa_bounds["J"] == (2, 4)
    assert proj.phase["V"]["end"] == 2            # 5 % 3 == 2 -> V/np boundary cuts a codon


def test_project_codons_frame_offset_shifts_residue_grid():
    # frame 1: residue r0 spans nt[1,4). V=[1,7) J=[7,13).
    proj = project_codons(13, 1, {"V": (1, 7), "J": (7, 13)})
    assert proj.n_residues == 4                    # (13 - 1)//3 == 4
    assert proj.provenance == ("V", "V", "J", "J")
    assert proj.aa_bounds["V"] == (0, 2)


def test_project_codons_d_region_and_np_gaps():
    # V=[0,6) np1=[6,9) D=[9,15) np2=[15,18) J=[18,24), frame 0.
    proj = project_codons(24, 0, {"V": (0, 6), "D": (9, 15), "J": (18, 24)})
    # residues: r0,r1 V ; r2 nt[6,9) np1 -> None ; r3 D ; r4 D ; r5 nt[15,18) np2 -> None ; r6,r7 J
    assert proj.provenance == ("V", "V", None, "D", "D", None, "J", "J")
    assert proj.aa_bounds["D"] == (3, 5)           # 2 fully-owned D residues


@pytest.mark.parametrize("fo", [0, 1, 2])
@pytest.mark.parametrize("vend", [6, 7, 8])
def test_project_codons_phase_matches_boundary_offset(fo, vend):
    proj = project_codons(30, fo, {"V": (fo, vend), "J": (vend, 30)})
    assert proj.phase["V"]["end"] == (vend - fo) % 3
    # a residue is owned by V iff its whole codon is inside [fo, vend)
    for r, owner in enumerate(proj.provenance):
        a = fo + 3 * r
        if a + 3 <= vend:
            assert owner == "V"
