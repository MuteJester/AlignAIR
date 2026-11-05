"""The test chip: AlignAIR's trained human IGH chip (``chip-v1``, 64-channel trunk), as the product's integration
tests predict with it (``tests.alignair.predict.golden_rows``), and its AIRR rows on the golden reads. CI never
rebuilds it; the file is the fixture.

It is the trained model ``chipw64-flanks-5272fc63-cool`` (SHA-256 ``SOURCE_SHA256``, 300k + 20k steps on GenAIRR's
HUMAN_IGH_OGRDB; dev suites and real-data arm in ``Notes and Plans/Architecture``) converted to an inference model
file by ``alignair.model_file.convert`` (weights and reference byte for byte). A briefly trained tiny chip was tried
first: at 1,500 steps it tied dozens of V alleles per read and assembled most rows only partially, too weak for
tests of the AIRR assembly.

Rebuild only for a deliberate change of the test model: ``python -m tests.alignair.fixtures.build_test_chip SOURCE``
(the frozen GenAIRR and ``src`` on PYTHONPATH), SOURCE the trained model file.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEST_CHIP = HERE / "test_chip.alignair"
ROWS = HERE / "test_chip_rows.json"
SOURCE_SHA256 = "f5ba8ed0122f33f7637f1e38b16f8f70b0be211c087e4dcfb3369e8aa932dbc8"


def build_test_chip(source: str, path: Path = TEST_CHIP) -> Path:
    """Convert the trained model ``source`` (checked against ``SOURCE_SHA256``) into ``path``."""
    from alignair.model_file.convert import convert_lab_chip
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"{source} is not the test chip's source model (SHA-256 {digest}, expected {SOURCE_SHA256})")
    convert_lab_chip(source, str(path))
    return path


def write_rows(model_path: Path = TEST_CHIP, rows_path: Path = ROWS) -> int:
    """Predict the golden reads with the test chip and write ``{"reads", "rows"}``; the number of reads."""
    from alignair.model_file import load_model
    from tests.alignair.predict.golden_rows import GOLDEN_READS, predict_rows
    reads = json.loads(GOLDEN_READS.read_text())["reads"]
    loaded = load_model(str(model_path))
    rows = predict_rows(loaded.model.eval(), loaded.reference, reads)
    rows_path.write_text(json.dumps({"reads": reads, "rows": rows}, sort_keys=True, indent=1))
    return len(reads)


if __name__ == "__main__":
    print(build_test_chip(sys.argv[1]), write_rows(), "reads")
