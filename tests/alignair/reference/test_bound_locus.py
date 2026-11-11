"""A reference of another locus bound to a one-locus model: binding stays allowed, the bound reference keeps its own
locus (so the records' ``locus`` comes from the alleles called, not the model card), and one warning says the model
was trained on another locus. A reference of the model's own locus, or one whose names show no locus, binds silently."""
import warnings

import GenAIRR.data as gd
import pytest

from alignair.reference import LocusMismatchWarning
from alignair.reference.anchors import annotate
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component


@pytest.fixture(scope="module")
def igh():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def test_a_trb_reference_bound_to_an_igh_model_keeps_its_locus_and_warns(igh):
    trb = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRB_IMGT)
    with pytest.warns(LocusMismatchWarning, match=r"TRB.*trained on IGH") as caught:
        bound = annotate(trb, igh)
    assert len(caught) == 1
    assert bound.locus_names() == ("TRB",)
    assert bound.loci[0].chain_type == "TCR_BETA"
    assert bound.gene("V").names == trb.gene("V").names


def test_a_reference_of_the_models_own_locus_binds_without_a_warning(igh):
    v = igh.gene("V")
    novel = v.sequences[0][:100] + ("A" if v.sequences[0][100] != "A" else "C") + v.sequences[0][101:]
    user = ReferenceSet.from_genotype({"V": {v.names[0]: v.sequences[0], "IGHVF1-G1*99": novel},
                                       "D": {"IGHD1-1*01": "GGTACAACTGGAACGAC"}, "J": {"IGHJ4*02": igh.gene("J").sequences[3]}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert annotate(user, igh).locus_names() == ("IGH",)


def test_a_trd_reference_whose_trav_dv_names_show_tra_binds_to_a_trd_model_without_a_warning():
    trd = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRD_IMGT)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert annotate(trd, trd).locus_names() == ("TRD",)


def test_a_reference_whose_names_show_no_locus_takes_the_models_without_a_warning(igh):
    v = igh.gene("V")
    user = ReferenceSet.from_genotype({"V": {"MYV1*01": v.sequences[0]}, "D": {"MYD1*01": "GGTACAACTGGAACGAC"},
                                       "J": {"MYJ1*01": igh.gene("J").sequences[3]}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert annotate(user, igh).locus_names() == ("IGH",)


def test_the_records_of_a_trb_reference_on_an_igh_model_say_trb():
    from alignair import Aligner
    from tests.alignair.predict.golden_rows import TEST_CHIP
    aligner = Aligner.from_pretrained(str(TEST_CHIP), device="cpu")
    ref = aligner.reference
    renamed = {g: {n.replace("IGH", "TRB"): s for n, s in zip(ref.gene(g).names, ref.gene(g).sequences)} for g in "VDJ"}
    trb = ReferenceSet.from_genotype(renamed)
    read = "".join(s for s in (ref.gene("V").sequences[5], "GGGAAA", ref.gene("D").sequences[2], "CCT",
                               ref.gene("J").sequences[3]))
    with pytest.warns(LocusMismatchWarning):
        bound = aligner.bound_to(trb)
    (record,) = bound.predict([read]).records
    assert record["locus"] == "TRB"
    assert record["v_call"].startswith("TRBV")


def test_predict_with_a_trb_reference_on_an_igh_model_writes_trb_rows_and_one_warning_line(tmp_path, capsys):
    import csv

    from alignair.cli.main import main
    from alignair.model_file import load_model, serialize
    from tests.alignair.predict.golden_rows import TEST_CHIP
    ref = load_model(str(TEST_CHIP)).reference
    fasta = serialize.reference_fasta(ref).replace(">IGH", ">TRB")
    (tmp_path / "trb.fasta").write_text(fasta)
    read = ref.gene("V").sequences[5] + "GGGAAA" + ref.gene("D").sequences[2] + "CCT" + ref.gene("J").sequences[3]
    (tmp_path / "r.fasta").write_text(f">r\n{read}\n")
    assert main(["predict", "--model", str(TEST_CHIP), "--input", str(tmp_path / "r.fasta"), "--out",
                 str(tmp_path / "o.tsv"), "--reference", str(tmp_path / "trb.fasta"), "--device", "cpu",
                 "--offline", "--quiet", "--no-run-metadata"]) == 0
    err = capsys.readouterr().err
    assert err.count("warning: the bound reference is TRB, but the model was trained on IGH") == 1
    assert "LocusMismatchWarning" not in err
    with open(tmp_path / "o.tsv", newline="") as handle:
        (row,) = list(csv.DictReader(handle, delimiter="\t"))
    assert row["locus"] == "TRB"


def test_a_reference_without_the_models_d_is_refused_with_a_hint():
    from alignair import Aligner
    from alignair.errors import ConfigurationError
    from tests.alignair.predict.golden_rows import TEST_CHIP
    igk = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB)
    with pytest.raises(ConfigurationError, match="no D alleles") as caught:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LocusMismatchWarning)
            Aligner.from_pretrained(str(TEST_CHIP), device="cpu", reference=igk)
    assert caught.value.hint and "V, D and J" in caught.value.hint and "reference export" in caught.value.hint
