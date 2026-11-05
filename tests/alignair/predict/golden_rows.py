"""The test chip and the golden reads of the product's integration tests: the model (``tests/alignair/fixtures/
test_chip.alignair``, built by ``tests.alignair.fixtures.build_test_chip``), the reads (``fixtures/golden_reads.json``:
GenAIRR reads drawn once) and the chip's rows of them (``FIXTURE``, ``tests/alignair/fixtures/test_chip_rows.json``),
and short amplicon reads whose V the chip ties with sibling alleles (``TIE_READS``, ``fixtures/tie_reads.json``: the
first six of a GenAIRR draw whose complete rows list ties), for the tests of ties and of the genotype pass.

Redraw the reads only for a deliberate change of the read set: ``python -m tests.alignair.predict.golden_rows``
(frozen GenAIRR on PYTHONPATH), then rewrite the rows (``build_test_chip.write_rows``).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

GOLDEN_READS = Path(__file__).parent / "fixtures" / "golden_reads.json"
TIE_READS = Path(__file__).parent / "fixtures" / "tie_reads.json"
TEST_CHIP = Path(__file__).parents[1] / "fixtures" / "test_chip.alignair"
FIXTURE = Path(__file__).parents[1] / "fixtures" / "test_chip_rows.json"


def golden_model():
    """``(model, reference_set)``: the test chip, in eval mode on the CPU, bound to its embedded reference."""
    from alignair.model_file import load_model
    loaded = load_model(str(TEST_CHIP))
    return loaded.model.eval(), loaded.reference


def draw_reads() -> list[str]:
    """Full-length reads (the first with a V indel), J-anchored amplicon reads, a read longer than the window, and a
    reverse-complemented read."""
    import GenAIRR.data as gd
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    components = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)
    full = list(itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[1], 21), 40))
    indel = next(r["sequence"] for r in full if r["n_v_indels"])
    plain = [r["sequence"] for r in full if not r["n_indels"]][:4]
    amplicons = [r["sequence"] for r in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[4], 22), 3)]
    complement = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return [indel, *plain, *amplicons, (plain[0] + plain[1])[:700], plain[2].translate(complement)[::-1]]


def predict_rows(model, reference, reads) -> list[dict]:
    """The rows ``predict_sequences`` gives, as JSON gives them back (numbers and strings, sorted keys)."""
    from alignair.api import predict_sequences
    rows = predict_sequences(model, reference, reads, device="cpu", batch_size=8)
    return json.loads(json.dumps(rows, default=str, sort_keys=True))


if __name__ == "__main__":
    reads = draw_reads()
    GOLDEN_READS.write_text(json.dumps({"reads": reads}, indent=1) + "\n")
    print(GOLDEN_READS, len(reads), "reads")
