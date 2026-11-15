"""The exported bundle's model_card.md: the model's id and version, its reference by name and version (the dataconfig
metadata, e.g. OGRDB V8), allele counts, training steps, intended use, the genotype and reference-swap usage lines,
the validation metrics (a "pending" sentence when the export ran without validation) and pointers to the
repository's LICENSE and CITATION.cff."""
import GenAIRR.data as gd
import pytest

from alignair import model_file as mf
from alignair.core import prepare_model
from alignair.genairr import resolve_dataconfigs
from alignair.reference import ReferenceSet
from alignair.train.build import _write_card, export_bundle

pytestmark = pytest.mark.integration


@pytest.fixture
def card(tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=128, embed_dim=4,
                          trunk_channels=8)
    ckpt = tmp_path / "model.alignair"
    mf.save_model(str(ckpt), setup.build_model().eval(), dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 3, "batch_size": 2})
    export_bundle(str(ckpt), resolve_dataconfigs(["HUMAN_IGH_OGRDB"]), str(tmp_path / "bundle"),
                  model_id="alignair4-igh-human", model_version="0.1.0", validate=False)
    return (tmp_path / "bundle" / "model_card.md").read_text()


def test_the_card_names_the_model_and_its_reference_by_name_and_version(card):
    assert "alignair4-igh-human 0.1.0" in card
    assert "HUMAN_IGH_OGRDB" in card and "OGRDB V8" in card and "Human" in card
    assert "alleles: V=198, D=33, J=7" in card and "training steps: 3" in card


def test_the_card_gives_intended_use_and_the_genotype_and_reference_swap_usage(card):
    assert "## Intended use" in card
    for line in ("--genotype donor.yaml", "--infer-genotype", "--reference germline.fasta"):
        assert line in card


def test_without_validation_the_card_says_the_metrics_are_pending_and_carries_no_internal_marker(card):
    assert "## Validation" in card and "Validation metrics are pending" in card
    assert "PENDING" not in card and "<!--" not in card


def test_the_card_points_at_the_repositorys_license_and_citation(card):
    assert "LICENSE" in card and "CITATION.cff" in card


def test_with_validation_the_metrics_are_listed_and_no_marker(tmp_path):
    manifest = {"loci": [{"locus": "IGH"}], "genes": {"V": {"n": 2}, "J": {"n": 1}}}
    path = tmp_path / "card.md"
    _write_card(str(path), {"steps": 5}, manifest, {"v_allele_top1": 0.5},
                dcs=resolve_dataconfigs(["HUMAN_IGH_OGRDB"]))
    text = path.read_text()
    assert "- v_allele_top1: 0.5" in text and "pending" not in text
