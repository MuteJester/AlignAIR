"""``python -m alignair_lab convert`` on a lab chip model file: a ``chip-v1`` inference file with the same weights and
reference, the lab's trunk width as ``trunk_channels``, loadable and giving the same rows without the lab; an AlignAIR 3
checkpoint and any other model file are refused."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from alignair import model_file as mf
from alignair.core import prepare_model
from alignair.errors import ConfigurationError
from alignair.model_file import container, read_metadata
from alignair.reference import ReferenceSet
from alignair_assay.diagnostics.golden import compare_rows, predict_rows
from alignair_lab.convert import LAB_CHIPS, convert_lab_chip, run_convert
from tests.alignair.model_file.conftest import write_v3_header_file

pytestmark = pytest.mark.integration


# The sources a subprocess imports, first on its PYTHONPATH: alignair_lab and alignair_assay are never installed, and
# an installed alignair may be.
SOURCE = str(Path(__file__).resolve().parents[2] / "src")


def _source_env(env: dict) -> dict:
    return {**env, "PYTHONPATH": os.pathsep.join(filter(None, [SOURCE, env.get("PYTHONPATH")]))}


# Settings of AlignAIR 3's configuration a lab chip file's configuration carried (the converter drops them).
LAB_ERA_SETTINGS = {"filters": 128, "block_out": 576, "latent_size_factor": 2, "classification_mode": "multilabel",
                    "boundary_mode": "half_open", "use_meta_heads": False, "chip_inputs": True}


def save_toy_lab_chip(path, lab_name: str, *, chip_copy=None) -> str:
    """A random-init chip on HUMAN_IGH_OGRDB as the lab's training runs saved one: with its training sections, and a
    configuration naming ``lab_name`` with AlignAIR 3's settings (``LAB_ERA_SETTINGS``) and no ``trunk_channels``
    (the lab name held the width): the file the converter reads. ``chip_copy``: also keep the same model saved as the
    chip-v1 file it was before its configuration was rewritten."""
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.model_file import save_model
    from alignair.reference.reference_set import ReferenceSet
    torch.manual_seed(0)
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), trunk_channels=LAB_CHIPS[lab_name])
    save_model(str(path), setup.build_model().eval(), setup.reference, dataconfigs=[gd.HUMAN_IGH_OGRDB],
               training={"steps": 7, "batch_size": 2}, include_trusted_pickle=True)
    if chip_copy is not None:
        shutil.copyfile(path, chip_copy)
    header = container.read_header(str(path))
    config = json.loads(container.read_section(str(path), "config"))
    config = {**{k: v for k, v in config.items() if k != "trunk_channels"}, "architecture": lab_name,
              **LAB_ERA_SETTINGS}
    sections, formats = {}, {}
    for name, section in header["sections"].items():
        sections[name] = (json.dumps(config).encode() if name == "config" else container.read_section(str(path), name),
                          section["codec"])
        formats[name] = section.get("format", "bytes")
    rewritten = {key: value for key, value in header.items() if not key.startswith("_") and key != "sections"}
    container.write_container(str(path), {**rewritten, "_formats": formats}, sections)
    return str(path)


def rows_in_clean_process(path, reads: list[str]) -> list[dict]:
    """``golden.predict_rows(path, reads)`` in a subprocess whose environment lacks ``ALIGNAIR_ARCHITECTURE_MODULES``
    (the lab never registered), as JSON rows."""
    env = _source_env({key: value for key, value in os.environ.items() if key != "ALIGNAIR_ARCHITECTURE_MODULES"})
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
    from alignair.train.records.streams import CurriculumStreamPolicy, stream_records
    component = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)[1]
    return [record["sequence"] for record in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, component, 21), 3)]


def _reference():
    import GenAIRR.data as gd
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def _model():
    return prepare_model(_reference(), max_seq_length=128, embed_dim=4, trunk_channels=8).build_model().eval()


def _v3_pt(tmp_path):
    path = str(tmp_path / "legacy.pt")
    torch.save({"config": {"v_allele_count": 2, "j_allele_count": 2}, "model": {}, "step": 7}, path)
    return path


@pytest.mark.parametrize("lab_name, channels", [("lab-chipw64-v1", 64), ("lab-chip-v1", 128)])
def test_a_converted_lab_chip_is_chip_v1_and_predicts_the_same_rows(tmp_path, lab_name, channels, reads):
    assert LAB_CHIPS == {"lab-chip-v1": 128, "lab-chipw64-v1": 64}
    src = save_toy_lab_chip(tmp_path / "lab.alignair", lab_name, chip_copy=tmp_path / "before.alignair")
    before = predict_rows(str(tmp_path / "before.alignair"), reads)
    summary = convert_lab_chip(src, str(tmp_path / "v4.alignair"))
    assert summary["from"] == lab_name and summary["to"] == "chip-v1" and summary["trunk_channels"] == channels
    after = rows_in_clean_process(tmp_path / "v4.alignair", reads)
    assert len(after) == len(before) and all(compare_rows(a, b) == [] for a, b in zip(before, after))
    header = read_metadata(str(tmp_path / "v4.alignair"))
    assert not any(name.startswith(("dataconfig/", "train_state")) for name in header["sections"])
    assert header["provenance"]["converted_from"]["architecture"] == lab_name
    assert header["reference"] == read_metadata(src)["reference"] | {
        "dataconfigs": [dict(d, section=None) for d in read_metadata(src)["reference"]["dataconfigs"]]}


def test_a_non_chip_model_file_is_refused(tmp_path):
    with pytest.raises(ConfigurationError, match="conv-v1"):
        convert_lab_chip(write_v3_header_file(tmp_path), str(tmp_path / "out.alignair"))


def test_the_cli_converts_a_lab_chip_without_trust_pickle(tmp_path):
    src = save_toy_lab_chip(tmp_path / "lab.alignair", "lab-chipw64-v1")
    out = subprocess.run([sys.executable, "-m", "alignair_lab", "convert", src, str(tmp_path / "v4.alignair")],
                         capture_output=True, text=True, env=_source_env(dict(os.environ)))
    assert out.returncode == 0, out.stderr
    assert "chip-v1" in out.stdout and read_metadata(str(tmp_path / "v4.alignair"))["sections"]


def test_the_cli_says_an_alignair_3_model_file_is_one(tmp_path, capsys):
    rc = run_convert(Namespace(src=write_v3_header_file(tmp_path), dst=str(tmp_path / "out.alignair")))
    err = capsys.readouterr().err
    assert rc != 0 and "AlignAIR 3 model" in err and "alignair<4" in err and "not a lab chip" not in err
    assert not (tmp_path / "out.alignair").exists()


def test_convert_refuses_a_pt_checkpoint(tmp_path, capsys):
    rc = run_convert(Namespace(src=_v3_pt(tmp_path), dst=str(tmp_path / "out.alignair")))
    assert rc != 0 and "alignair<4" in capsys.readouterr().err
    assert not (tmp_path / "out.alignair").exists()


def test_convert_refuses_a_model_file_that_is_not_a_lab_chip(tmp_path, capsys):
    p = str(tmp_path / "m.alignair")
    mf.save_model(p, _model(), _reference(), dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)
    rc = run_convert(Namespace(src=p, dst=str(tmp_path / "out.alignair")))
    assert rc != 0 and "not a lab chip model file" in capsys.readouterr().err
    assert not (tmp_path / "out.alignair").exists()
