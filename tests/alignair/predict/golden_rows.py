"""The test chip and the golden reads of the product's integration tests: the model (``tests/alignair/fixtures/
test_chip.alignair``, built by ``tests.alignair.fixtures.build_test_chip``), the reads (``fixtures/golden_reads.json``:
GenAIRR reads drawn once) and the chip's rows of them (``FIXTURE``, ``tests/alignair/fixtures/test_chip_rows.json``),
and short amplicon reads whose V the chip ties with sibling alleles (``TIE_READS``, ``fixtures/tie_reads.json``: the
first six of a GenAIRR draw whose complete rows list ties), for the tests of ties and of the genotype pass.

Redraw the reads only for a deliberate change of the read set: ``python -m tests.alignair.predict.golden_rows``
(frozen GenAIRR on PYTHONPATH), then rewrite the rows (``build_test_chip.write_rows``).

The chip's continuous outputs are float32 sums whose order follows the CPU's kernels and the thread count, so the
recorded rows are compared with ``assert_rows_match``, not ``==``: across 1 to 16 threads the indel count moves by up
to 5e-6 and the mutation rate by up to 1e-6, and an allele tied with another may take its place.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

GOLDEN_READS = Path(__file__).parent / "fixtures" / "golden_reads.json"
TIE_READS = Path(__file__).parent / "fixtures" / "tie_reads.json"
TEST_CHIP = Path(__file__).parents[1] / "fixtures" / "test_chip.alignair"
FIXTURE = Path(__file__).parents[1] / "fixtures" / "test_chip_rows.json"
# How closely a recorded float must be met: well above the spread of the summation order (see the module docstring),
# far below any change of the model or of the decoding.
FLOAT_TOLERANCE = {"rel": 1e-4, "abs": 2e-5}


def golden_model():
    """``(model, reference_set)``: the test chip, in eval mode on the CPU, bound to its embedded reference."""
    from alignair.model_file import load_model
    loaded = load_model(str(TEST_CHIP))
    return loaded.model.eval(), loaded.reference


def draw_reads() -> list[str]:
    """Full-length reads (the first with a V indel), J-anchored amplicon reads, a read longer than the window, and a
    reverse-complemented read."""
    import GenAIRR.data as gd
    from alignair.train.records.streams import CurriculumStreamPolicy, stream_records
    components = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)
    full = list(itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[1], 21), 40))
    indel = next(r["sequence"] for r in full if r["n_v_indels"])
    plain = [r["sequence"] for r in full if not r["n_indels"]][:4]
    amplicons = [r["sequence"] for r in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[4], 22), 3)]
    complement = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return [indel, *plain, *amplicons, (plain[0] + plain[1])[:700], plain[2].translate(complement)[::-1]]


def predict_rows(model, reference, reads) -> list[dict]:
    """The rows ``Aligner.predict`` gives, as JSON gives them back (numbers and strings, sorted keys)."""
    from alignair import Aligner
    rows = Aligner.from_model(model, reference, device="cpu").predict(reads, batch_size=8).records
    return json.loads(json.dumps(rows, default=str, sort_keys=True))


def assert_rows_match(rows: list[dict], expected: list[dict]) -> None:
    """``rows`` are the recorded ``expected`` rows: the same rows with the same keys, every value exact (calls, sets
    of tied calls, junctions, coordinates, sequences, alignments, flags) but a float, which meets
    ``FLOAT_TOLERANCE``, and a gene's candidate list (``<gene>_calls`` with its ``<gene>_likelihoods``), which meets
    the near-tie rule.

    The near-tie rule. A gene's candidates are its alleles scoring at least the threshold, best first (ties in the
    bank's order), at most the cap of them (``predict.threshold``, with ``PredictConfig``'s cap). Scores that meet
    each other within ``FLOAT_TOLERANCE`` are a near-tie: on another machine their alleles may come in the other order,
    and when the list is full its last places may go to any allele tied with them, an unlisted one too. So the
    likelihoods meet the recorded ones place by place; the names are the recorded ones place by place, except in a run
    of near-tied places, where they are the recorded names in any order, and in the run that ends a full list, where
    they need only be distinct alleles (of the run's score: the likelihoods say so). A score at the threshold itself
    is not tolerated: none of the golden reads' scores lies within 0.1 of it."""
    assert len(rows) == len(expected)
    for index, (row, want) in enumerate(zip(rows, expected)):
        assert sorted(row) == sorted(want), (index, sorted(set(row) ^ set(want)))
        for key, value in want.items():
            where = (index, key, row[key], value)
            if key.endswith("_likelihoods"):
                continue                                         # with its gene's _calls
            if key.endswith("_calls"):
                likelihoods = key[:-len("_calls")] + "_likelihoods"
                _assert_candidates(row[key], row[likelihoods], value, want[likelihoods], where)
            elif isinstance(value, float) and isinstance(row[key], (int, float)) and not isinstance(row[key], bool):
                assert row[key] == pytest.approx(value, **FLOAT_TOLERANCE), where
            else:
                assert row[key] == value, where


def _assert_candidates(names, likelihoods, want_names, want_likelihoods, where):
    """One gene's candidate list meets the recorded one by ``assert_rows_match``'s near-tie rule."""
    from alignair.predict.config import PredictConfig
    assert len(names) == len(likelihoods) == len(want_names) == len(want_likelihoods), where
    assert likelihoods == pytest.approx(want_likelihoods, **FLOAT_TOLERANCE), where
    start = 0
    while start < len(want_names):
        end = start + 1
        while end < len(want_names) and want_likelihoods[end] == pytest.approx(want_likelihoods[end - 1],
                                                                               **FLOAT_TOLERANCE):
            end += 1
        run = names[start:end]
        if end == len(want_names) == PredictConfig.cap:
            assert len(set(run)) == len(run) and not set(run) & set(names[:start]), where
        else:
            assert sorted(run) == sorted(want_names[start:end]), where
        start = end


if __name__ == "__main__":
    reads = draw_reads()
    GOLDEN_READS.write_text(json.dumps({"reads": reads}, indent=1) + "\n")
    print(GOLDEN_READS, len(reads), "reads")
