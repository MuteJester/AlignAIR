"""AlignAIR 4 reads chip models only: a model file of AlignAIR 3 (conv-v1) is refused with the version that reads it,
before anything is built, whether it is loaded or only read."""
import pytest

from alignair.errors import ModelError
from alignair.model_file import load_model, read_model_state

from .conftest import write_v3_header_file

pytestmark = pytest.mark.component


@pytest.mark.parametrize("read", [load_model, read_model_state])
def test_a_v3_model_file_is_refused_with_the_version_that_reads_it(tmp_path, read):
    with pytest.raises(ModelError, match=r"AlignAIR 3 model .*conv-v1.*alignair<4"):
        read(write_v3_header_file(tmp_path))


def test_a_configuration_with_alignair_3_settings_is_refused_with_the_version_that_reads_it():
    """A pre-release lab file (another architecture, AlignAIR 3's settings) is refused, pointing at alignair<4; the
    lab converts its own files."""
    import json

    from alignair.model_file.serialize import config_from_bytes
    config = {"v_allele_count": 2, "j_allele_count": 2, "architecture": "lab-chipw64-v1", "filters": 128}
    with pytest.raises(ModelError, match="written before AlignAIR 4") as caught:
        config_from_bytes(json.dumps(config).encode())
    assert "alignair<4" in caught.value.hint and caught.value.context["settings"] == ["filters"]
