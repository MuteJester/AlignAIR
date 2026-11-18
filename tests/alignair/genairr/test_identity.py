"""Model files and bundles record which GenAIRR made them: its determinism version and its source digest."""
import re

import pytest

import GenAIRR
import GenAIRR.data as gd
from alignair.core import prepare_model
from alignair.model_file import read_metadata, save_model
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component


def _identity_ok(record):
    return (record["genairr_determinism_version"] == GenAIRR.DETERMINISM_VERSION
            and re.fullmatch(r"[0-9a-f]{64}", record["genairr_source_sha256"]))


def test_a_saved_model_records_the_genairr_identity(tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=128, embed_dim=4,
                          trunk_channels=8)
    path = tmp_path / "model.alignair"
    save_model(path, setup.build_model(), setup.reference,
               dataconfigs=[gd.HUMAN_IGH_OGRDB], training={"steps": 0, "batch_size": 0},
               include_trusted_pickle=False)
    assert _identity_ok(read_metadata(path))     # beside alignair_version and torch_version


def test_a_bundle_manifest_records_the_genairr_identity(tmp_path):
    from alignair.train.bundle import _reference_manifest
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    model = tmp_path / "model.alignair"
    model.write_bytes(b"x")
    assert _identity_ok(_reference_manifest(reference, None, None, str(model))["versions"])
