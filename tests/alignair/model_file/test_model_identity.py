"""A model file's identity is its embedded reference and the two digests of it: a new file stores no allele counts, no
label schema and no one-value input or label space, and a file that stores them still loads."""
import contextlib
import io
import json

import GenAIRR.data as gd
import pytest

from alignair.cli.main import main
from alignair.core import prepare_model
from alignair.errors import ModelError
from alignair.model_file import (LoadedModel, ModelStateData, TrainingState, TrainingStateData, container,
                                 load_model, read_metadata, save_model, serialize)
from alignair.model_file.serialize import config_from_bytes
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.integration

DROPPED_CONFIG = {"v_allele_count", "j_allele_count", "d_allele_count", "input_space", "label_space"}
DROPPED_HEADER = {"allele_counts", "label_schema", "input_space", "label_space"}


@pytest.fixture(scope="module")
def reference():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def _save(path, reference, **options):
    model = prepare_model(reference, max_seq_length=128, embed_dim=4, trunk_channels=8).build_model()
    save_model(str(path), model, reference, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
               **options)
    return model


def test_a_new_file_stores_its_reference_and_its_digests_only(tmp_path, reference):
    _save(tmp_path / "m.alignair", reference)
    md = read_metadata(str(tmp_path / "m.alignair"))
    assert not DROPPED_HEADER & set(md["model"])
    assert not DROPPED_CONFIG & set(json.loads(container.read_section(str(tmp_path / "m.alignair"), "config")))
    assert md["reference"]["allele_order_sha256"] == serialize.allele_order_sha256(reference)
    assert md["reference"]["reference_fasta_sha256"] == serialize.reference_fasta_sha256(reference)


def test_the_file_embeds_the_reference_the_model_is_bound_to(tmp_path, reference):
    donor = reference.subset({"V": list(reference.gene("V").names)[:20]})
    _save(tmp_path / "m.alignair", donor)
    assert container.read_section(str(tmp_path / "m.alignair"), "reference_json") == serialize.reference_to_json(donor)
    assert len(load_model(str(tmp_path / "m.alignair")).reference.gene("V")) == 20


def test_a_file_storing_counts_spaces_and_a_label_schema_loads_as_the_same_model(tmp_path, reference):
    _save(tmp_path / "m.alignair", reference)
    path = str(tmp_path / "m.alignair")
    header = container.read_header(path)
    sections = {name: (container.read_section(path, name), meta["codec"]) for name, meta in header["sections"].items()}
    config = json.loads(sections["config"][0])
    config.update(v_allele_count=1, j_allele_count=1, d_allele_count=1, input_space="nucleotide",
                  label_space="alleles")
    sections["config"] = (json.dumps(config).encode(), sections["config"][1])
    old = {key: value for key, value in header.items() if not key.startswith("_") and key != "sections"}
    old["model"] = {**old["model"], "label_schema": {"schema_version": 1, "genes": {"v": ["not-an-allele"]}},
                    "allele_counts": {"v": 1, "d": 1, "j": 1}, "input_space": "nucleotide", "label_space": "alleles"}
    old["_formats"] = {name: meta.get("format", "bytes") for name, meta in header["sections"].items()}
    container.write_container(str(tmp_path / "old.alignair"), old, sections)
    loaded = load_model(str(tmp_path / "old.alignair"))
    assert loaded.config == load_model(path).config
    assert len(loaded.reference.gene("V")) == len(reference.gene("V"))


@pytest.mark.parametrize("field, value", [("input_space", "protein"), ("label_space", "allele_classes")])
def test_a_space_another_than_its_one_value_is_refused(field, value):
    with pytest.raises(ModelError, match=field):
        config_from_bytes(json.dumps({"architecture": "chip-v1", "has_d": True, field: value}).encode())


def test_info_counts_the_alleles_of_the_embedded_reference(tmp_path, reference):
    _save(tmp_path / "m.alignair", reference)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main(["info", str(tmp_path / "m.alignair")]) == 0
    v, d, j = (len(reference.gene(gene)) for gene in "VDJ")
    assert f"V={v} D={d} J={j}" in out.getvalue()


def test_a_card_cannot_set_a_structural_key(tmp_path, reference):
    _save(tmp_path / "m.alignair", reference, card={"config_schema_version": 99, "species": "human"})
    md = read_metadata(str(tmp_path / "m.alignair"))
    assert md["config_schema_version"] == 1 and md["species"] == "human"


def test_a_training_state_is_a_model_state_with_its_training_fields():
    assert issubclass(TrainingStateData, ModelStateData) and issubclass(TrainingState, LoadedModel)
