"""AA artifacts use the shared strict restoration boundary after integrity verification."""
import pytest

from alignair import load_aa_model
from alignair.errors import ModelIntegrityError
from alignair.model_file import container, serialize
from tests.support.protein_bundle import build_protein_bundle_setup, write_protein_bundle

pytestmark = pytest.mark.integration


def test_verified_aa_payload_with_incompatible_weights_has_shared_diagnostics(tmp_path):
    path = tmp_path / "protein.alignair"
    write_protein_bundle(build_protein_bundle_setup(), path)
    header = container.read_header(path)
    header.pop("_sections_base", None)
    sections = {name: (container.read_section(path, name), info["codec"])
                for name, info in header["sections"].items()}
    state = serialize.state_dict_from_bytes(sections["weights"][0])
    del state[next(iter(state))]
    weights = serialize.state_dict_to_bytes(state)
    sections["weights"] = (weights, sections["weights"][1])
    header["integrity"]["weights_sha256"] = container.sha256_hex(weights)
    container.write_container(path, header, sections)

    with pytest.raises(ModelIntegrityError, match="Missing key") as caught:
        load_aa_model(path, device="cpu")
    assert caught.value.context["stage"] == "model_restoration"
    assert caught.value.context["path"] == str(path)
    assert isinstance(caught.value.__cause__, RuntimeError)
