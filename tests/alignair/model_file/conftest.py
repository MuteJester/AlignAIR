"""Model files for the conversion tests: the toy chip saved under a lab name, a model file whose configuration names
conv-v1 (written without building conv-v1), and the rows a model file gives in a process without the lab."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from alignair.model_file import container, serialize


def save_toy_lab_chip(path, lab_name: str) -> str:
    """A random-init chip on HUMAN_IGH_OGRDB built under ``lab_name`` (the lab registered), saved with its training
    sections (as the lab's training runs saved theirs): the file the converter reads."""
    import GenAIRR.data as gd
    import alignair_lab.architectures  # noqa: F401  (registers the lab chips)
    from alignair.core import prepare_model
    from alignair.model_file import save_model
    from alignair.reference.reference_set import ReferenceSet
    torch.manual_seed(0)
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), architecture=lab_name,
                          boundary_mode="half_open", anchor_heads=True, d_presence_head=True, chip_inputs=True,
                          bank_orientation=True)
    save_model(str(path), setup.build_model().eval(), dataconfigs=[gd.HUMAN_IGH_OGRDB],
               training={"steps": 7, "batch_size": 2}, include_trusted_pickle=True)
    return str(path)


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


def rows_in_clean_process(path, reads: list[str]) -> list[dict]:
    """``golden.predict_rows(path, reads)`` in a subprocess whose environment lacks ``ALIGNAIR_ARCHITECTURE_MODULES``
    (the lab never registered), as JSON rows."""
    env = {key: value for key, value in os.environ.items() if key != "ALIGNAIR_ARCHITECTURE_MODULES"}
    code = ("import json, sys; from alignair_assay.diagnostics.golden import predict_rows; "
            "print(json.dumps(predict_rows(sys.argv[1], json.loads(sys.argv[2]))))")
    out = subprocess.run([sys.executable, "-c", code, str(path), json.dumps(reads)], capture_output=True, text=True,
                         env=env, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def reads() -> list[str]:
    """Three GenAIRR heavy-chain reads."""
    import itertools
    import GenAIRR.data as gd
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    component = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)[1]
    return [record["sequence"] for record in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, component, 21), 3)]
