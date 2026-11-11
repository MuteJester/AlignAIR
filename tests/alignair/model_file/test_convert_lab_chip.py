"""``alignair convert`` on a lab chip model file: a ``chip-v1`` inference file with the same weights and reference, the
lab's trunk width as ``trunk_channels``, loadable and giving the same rows without the lab; anything else refused."""
import subprocess
import sys

import pytest

from alignair.errors import ConfigurationError
from alignair.model_file import read_metadata
from alignair.model_file.convert import LAB_CHIPS, convert_lab_chip
from alignair_assay.diagnostics.golden import compare_rows, predict_rows

from .conftest import rows_in_clean_process, save_toy_lab_chip, write_v3_header_file

pytestmark = pytest.mark.integration


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


def test_a_lab_chip_file_is_refused_until_converted(tmp_path):
    from alignair.errors import ModelError
    from alignair.model_file import load_model
    with pytest.raises(ModelError, match="lab chip model file .*convert it"):
        load_model(save_toy_lab_chip(tmp_path / "lab.alignair", "lab-chipw64-v1"))


def test_a_non_chip_model_file_is_refused(tmp_path):
    with pytest.raises(ConfigurationError, match="conv-v1"):
        convert_lab_chip(write_v3_header_file(tmp_path), str(tmp_path / "out.alignair"))


def test_the_cli_converts_a_lab_chip_without_trust_pickle(tmp_path):
    src = save_toy_lab_chip(tmp_path / "lab.alignair", "lab-chipw64-v1")
    out = subprocess.run([sys.executable, "-c", "from alignair.cli.main import main; raise SystemExit(main())",
                          "convert", src, str(tmp_path / "v4.alignair")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "chip-v1" in out.stdout and read_metadata(str(tmp_path / "v4.alignair"))["sections"]


def test_the_cli_says_an_alignair_3_model_file_is_one(tmp_path, capsys):
    from argparse import Namespace

    from alignair.cli import convert
    rc = convert.run(Namespace(src=write_v3_header_file(tmp_path), dst=str(tmp_path / "out.alignair")))
    err = capsys.readouterr().err
    assert rc != 0 and "AlignAIR 3 model" in err and "alignair<4" in err and "not a lab chip" not in err
    assert not (tmp_path / "out.alignair").exists()
