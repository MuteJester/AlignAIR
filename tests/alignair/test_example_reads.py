"""``examples/reads.fasta``, the reads the README's first alignment uses, are what ``examples/make_example_reads.py``
writes: 20 clean, productive human IGH reads (the same file wherever GenAIRR draws the same records)."""
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.component

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _records(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    return list(zip(lines[0::2], lines[1::2]))


def test_the_example_reads_are_twenty_named_clean_reads():
    records = _records((EXAMPLES / "reads.fasta").read_text())
    assert [name for name, _ in records] == [f">read{i:02d}" for i in range(1, 21)]
    assert all(set(sequence) <= set("ACGT") and 300 <= len(sequence) <= 600 for _, sequence in records)


def test_the_example_reads_are_reproduced_by_their_script():
    spec = importlib.util.spec_from_file_location("make_example_reads", EXAMPLES / "make_example_reads.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.example_reads() == (EXAMPLES / "reads.fasta").read_text()
