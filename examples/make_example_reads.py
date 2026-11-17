"""Write ``examples/reads.fasta``, the reads the README's first alignment uses: 20 productive human IGH reads simulated
by GenAIRR from the HUMAN_IGH_OGRDB germline set (seed 7, light somatic hypermutation, no N bases), so a first run
shows typical calls. Run from the repository root: ``python examples/make_example_reads.py``."""
from __future__ import annotations

import itertools
from pathlib import Path

SEED = 7
COUNT = 20
OUT = Path(__file__).with_name("reads.fasta")


def example_reads(seed: int = SEED, count: int = COUNT) -> str:
    """``count`` productive IGH reads as FASTA text, named ``read01``, ``read02``, ..."""
    from alignair.genairr import resolve_dataconfig
    from alignair.genairr.experiment import build_experiment
    from alignair.train.records import Curriculum
    params = {**Curriculum().params(0.2), "productive_only": True, "invert_d_prob": 0.0, "ambiguous_count": (0, 0)}
    experiment = build_experiment(resolve_dataconfig("HUMAN_IGH_OGRDB"), params, allow_curatable=True)
    records = itertools.islice(experiment.stream_records(n=None, seed=seed), count)
    return "".join(f">read{i:02d}\n{str(record['sequence']).upper()}\n" for i, record in enumerate(records, 1))


if __name__ == "__main__":
    OUT.write_text(example_reads())
    print(f"wrote {OUT}")
