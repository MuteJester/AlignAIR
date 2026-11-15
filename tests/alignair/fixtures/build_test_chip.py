"""The test chip: a ``chip-v1`` trained for the package's tests alone, as the product's integration tests predict with
it (``tests.alignair.predict.golden_rows``), its AIRR rows on the golden reads, and the tie reads (short amplicons
whose V it ties with sibling alleles). CI never rebuilds it; the file is the fixture.

It is not a product model. It was trained by the lab recipe (``alignair_lab.recipe.training_config``) on chip-v1 with
its defaults (64-channel trunk, anchor and D-presence heads, bank orientation): HUMAN_IGH_OGRDB on GenAIRR 2.4.0, batch
64, library flanks, seed 1, 28,000 steps under the cosine schedule from 3e-4 to 3e-6 (``Notes and Plans/Architecture/
2026-11-06-test-chip-plan.md``). The training checkpoint (SHA-256 ``SOURCE_SHA256``) is exported by the product's
``alignair.train.bundle.export_bundle``: pickle-free, and naming neither the host nor the user it was made on. A
briefly trained tiny chip was tried before the trained product chip served as the fixture: at 1,500 steps it tied
dozens of V alleles per read and assembled most rows only partially, too weak for tests of the AIRR assembly.

Rebuild only for a deliberate change of the test model: ``python -m tests.alignair.fixtures.build_test_chip SOURCE``
(the frozen GenAIRR and ``src`` on PYTHONPATH), SOURCE the training checkpoint; it writes the model, the rows and the
tie reads.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
TEST_CHIP = HERE / "test_chip.alignair"
ROWS = HERE / "test_chip_rows.json"
TIE_READS = HERE.parent / "predict" / "fixtures" / "tie_reads.json"
SOURCE_SHA256 = "6262cdefa43683b16e0ba0ffbfbea507c3b49d2d2099bc002d83379d9e3130a7"
TIE_COMPONENTS, TIE_SEED, TIE_DRAW, TIE_COUNT = (4, 5, 6, 1), 41, 80, 6


def build_test_chip(source: str, path: Path = TEST_CHIP) -> Path:
    """Export the training checkpoint ``source`` (checked against ``SOURCE_SHA256``) as the pickle-free ``path``."""
    from alignair.genairr import resolve_dataconfigs
    from alignair.train.bundle import export_bundle
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"{source} is not the test chip's source model (SHA-256 {digest}, expected {SOURCE_SHA256})")
    with tempfile.TemporaryDirectory() as scratch:
        model = export_bundle(source, resolve_dataconfigs(["HUMAN_IGH_OGRDB"]), str(Path(scratch) / "bundle"),
                              model_id="alignair-test-chip", validate=False,
                              description="AlignAIR's test chip: chip-v1 trained briefly (28k steps, seed 1) on "
                                          "HUMAN_IGH_OGRDB for the package's integration tests; not a product model.")
        shutil.copyfile(model, path)
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


def write_tie_reads(model_path: Path = TEST_CHIP, path: Path = TIE_READS) -> int:
    """The first ``TIE_COUNT`` GenAIRR reads (amplicon components first, then full length) whose complete row's V call
    lists ties under the test chip, written as ``{"reads"}``; the number written."""
    import GenAIRR.data as gd
    from alignair.api import predict_sequences
    from alignair.model_file import load_model
    from alignair.train.records.streams import CurriculumStreamPolicy, stream_records
    loaded = load_model(str(model_path))
    components = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)
    found = []
    for index in TIE_COMPONENTS:
        reads = [r["sequence"] for r in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[index],
                                                                        TIE_SEED), TIE_DRAW)]
        rows = predict_sequences(loaded.model.eval(), loaded.reference, reads, device="cpu", batch_size=16)
        found += [read for read, row in zip(reads, rows)
                  if "," in (row["v_call"] or "") and row.get("airr_assembly_status") == "complete"]
        if len(found) >= TIE_COUNT:
            break
    path.write_text(json.dumps({"reads": found[:TIE_COUNT]}, indent=1))
    return len(found[:TIE_COUNT])


if __name__ == "__main__":
    print(build_test_chip(sys.argv[1]), write_rows(), "reads,", write_tie_reads(), "tie reads")
