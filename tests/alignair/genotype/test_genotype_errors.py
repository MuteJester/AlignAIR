"""A genotype file is YAML or JSON keyed by segment (V, D, J), each listing allele names. A FASTA, any other
unparseable file, a file that is not such a mapping, or one keyed by gene names is refused with a ConfigurationError
whose hint gives the format, never an internal error."""
import GenAIRR.data as gd
import pytest

from alignair.errors import ConfigurationError
from alignair.genotype import genotype_bank
from alignair.genotype.files import load_genotype
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.component


@pytest.fixture(scope="module")
def igh():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


@pytest.mark.parametrize("name, text", [
    ("donor.fasta", ">IGHVF1-G3*02\nCAGGTCACCTTG\n>IGHJ4*02\nACTACTTTGAC\n"),
    ("donor.yaml", "V: [IGHVF1-G3*02\n"),
    ("donor.json", "{\"V\": [\"IGHVF1-G3*02\"\n"),
])
def test_an_unparseable_genotype_file_is_a_configuration_error_with_the_format_hint(tmp_path, igh, name, text):
    path = tmp_path / name
    path.write_text(text)
    with pytest.raises(ConfigurationError, match="not a YAML or JSON genotype") as caught:
        load_genotype(str(path))
    assert "V, D, J" in caught.value.hint and "--reference" in caught.value.hint
    assert caught.value.context["path"] == str(path)


@pytest.mark.parametrize("text", ["- IGHVF1-G3*02\n- IGHJ4*02\n", "just text\n", "V: IGHVF1-G3*02\n"])
def test_a_genotype_that_is_not_a_segment_to_alleles_mapping_is_refused(tmp_path, igh, text):
    path = tmp_path / "donor.yaml"
    path.write_text(text)
    with pytest.raises(ConfigurationError) as caught:
        load_genotype(str(path))
    assert "V, D, J" in caught.value.hint


def test_a_genotype_keyed_by_gene_names_says_so(tmp_path, igh):
    path = tmp_path / "donor.yaml"
    path.write_text("IGHVF1-G3:\n  - IGHVF1-G3*02\nIGHVF6-G21:\n  - IGHVF6-G21*07\n")
    with pytest.raises(ConfigurationError, match=r"gene names \(IGHVF1-G3, IGHVF6-G21\)") as caught:
        load_genotype(str(path))
    assert "V, D, J" in caught.value.hint


def test_genotype_bank_refuses_gene_name_keys_and_unknown_keys_alike(igh):
    with pytest.raises(ConfigurationError, match=r"gene names \(IGHVF1-G3\)"):
        genotype_bank(igh, {"IGHVF1-G3": ["IGHVF1-G3*02"]})
    with pytest.raises(ConfigurationError, match=r"keys are segments"):
        genotype_bank(igh, {"heavy": ["IGHVF1-G3*02"]})


def test_segment_keys_in_any_case_still_load(tmp_path, igh):
    path = tmp_path / "donor.yaml"
    path.write_text("v: [IGHVF1-G3*02]\nJ: [IGHJ4*02]\n")
    genotype = load_genotype(str(path))
    assert genotype == {"v": {"IGHVF1-G3*02"}, "j": {"IGHJ4*02"}}
    assert len(genotype_bank(igh, genotype).gene("V")) == 1


def test_predict_with_a_fasta_genotype_exits_as_a_configuration_error(tmp_path, capsys):
    from alignair.cli.main import main
    from tests.alignair.predict.golden_rows import TEST_CHIP
    (tmp_path / "r.fasta").write_text(">r\nCAGGTGCAGCTGGTGCAGTCTGGG\n")
    (tmp_path / "donor.fasta").write_text(">IGHVF1-G3*02\nCAGGTCACCTTG\n")
    status = main(["predict", "--model", str(TEST_CHIP), "--input", str(tmp_path / "r.fasta"), "--out",
                   str(tmp_path / "o.tsv"), "--genotype", str(tmp_path / "donor.fasta"), "--device", "cpu",
                   "--offline", "--quiet", "--no-run-metadata"])
    err = capsys.readouterr().err
    assert status == 1 and "ALIGNAIR_CONFIG_INVALID" in err and "INTERNAL" not in err
    assert "V, D, J" in err
