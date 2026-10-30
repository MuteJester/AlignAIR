"""The conv-v1 model and reads of the golden AIRR rows (``fixtures/conv_v1_rows.json``), shared by the test and the
script that wrote the fixture.

The model is Task 1's golden conv-v1 (``tests/alignair/core/test_trunk_hooks.py``: HUMAN_IGH_OGRDB, random init from
seed 0) with its boundary heads shaped so every read's V, D and J fall on spans of real size, so the germline reader
and the AIRR assembly run on real alignments. The reads are GenAIRR reads drawn once and stored in the fixture.

Rewrite the fixture only for a deliberate change of conv-v1's rows:
``python -m tests.alignair.predict.golden_rows`` (frozen GenAIRR on PYTHONPATH).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import torch

FIXTURE = Path(__file__).parent / "fixtures" / "conv_v1_rows.json"
# forward-frame spans the shaped boundary heads give every read (0-based, end-exclusive)
SPANS = {"v": (0, 290), "d": (300, 316), "j": (322, 360)}


def golden_model():
    """``(model, reference_set)``: the shaped golden conv-v1, in eval mode on the CPU."""
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.reference.reference_set import ReferenceSet
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    torch.manual_seed(0)
    model = prepare_model(reference).build_model().eval()
    with torch.no_grad():
        for g, (start, end) in SPANS.items():
            for head, position in ((model.branches[g].start_head, start), (model.branches[g].end_head, end)):
                head.bias.zero_()
                head.bias[position] = 30.0
    return model, reference


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
    model, reference = golden_model()
    reads = draw_reads()
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps({"reads": reads, "rows": predict_rows(model, reference, reads)}, sort_keys=True,
                                  indent=1))
    print(FIXTURE, len(reads), "reads")
