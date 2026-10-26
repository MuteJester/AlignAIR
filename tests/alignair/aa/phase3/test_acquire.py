import pytest

from alignair.aa.phase3.cohort import load_crosswalk, load_hydrated

pytestmark = pytest.mark.component


def test_crosswalk_requires_citation(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("alias\timgt_family\timgt_gene\tcitation\nVH subgroup III\tIGHV3\t\tScaviner1999\n")
    cw = load_crosswalk(str(p))
    assert cw["VH subgroup III"] == ("IGHV3", "", "Scaviner1999")


def test_crosswalk_rejects_uncited_row(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("alias\timgt_family\timgt_gene\tcitation\nVH subgroup III\tIGHV3\t\t\n")
    with pytest.raises(ValueError):
        load_crosswalk(str(p))


def test_load_hydrated_maps_id_to_sequence(tmp_path):
    p = tmp_path / "h.tsv"
    p.write_text("exact_vh_id\tvh_aa\nabc123\tEVQLVES\n")
    assert load_hydrated(str(p)) == {"abc123": "EVQLVES"}
