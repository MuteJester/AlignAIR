"""A model file whose configuration names conv-v1 (written without building conv-v1), for the AlignAIR 3 refusal
tests."""
from __future__ import annotations

import json


from alignair.model_file import container, serialize


def write_v3_header_file(tmp_path) -> str:
    """A model container whose configuration names architecture ``conv-v1``: an AlignAIR 3 model file's header, with
    no conv-v1 code involved (its weights section is empty)."""
    path = tmp_path / "v3.alignair"
    config = {"v_allele_count": 2, "j_allele_count": 2, "d_allele_count": 2, "architecture": "conv-v1"}
    header = {"format_version": container.MAJOR_VERSION, "model_format_version": container.MAJOR_VERSION,
              "model_class": "AlignAIR", "config_schema_version": 1, "model": {}, "reference": {},
              "_formats": {"config": "json", "weights": "safetensors"}}
    sections = {"config": (json.dumps(config).encode(), "zlib"),
                "weights": (serialize.state_dict_to_bytes({}), "zlib")}
    container.write_container(str(path), header, sections)
    return str(path)
