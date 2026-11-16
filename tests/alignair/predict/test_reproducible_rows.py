"""A read's row does not depend on the reads of other lengths it is predicted with: the same reads, predicted in two
files whose other reads fall in other width buckets (batch size and threads fixed), get identical rows, every float
included. Reads of the same bucket share batches, and how many share one can still move its last float bits."""
import json

import pytest
import torch

from tests.alignair.predict.golden_rows import FIXTURE, golden_model

pytestmark = pytest.mark.integration


def test_the_same_read_in_two_files_gets_the_same_row():
    from alignair import Aligner
    from alignair.core.boundaries import width_bucket
    torch.set_num_threads(8)                        # the harness's two threads hide width effects (it restores them)
    model, reference = golden_model()
    reads = json.loads(FIXTURE.read_text())["reads"]
    window = model.cfg.max_seq_length - 1
    bucket = lambda read: width_bucket(min(len(read), window) + 1, model.cfg.max_seq_length)
    short = [read[:60] for read in reads[:5]]
    long_ = [(read + read)[:150] for read in reads[:5]] + [(read + read)[:400] for read in reads[:5]]
    assert not {bucket(r) for r in reads} & {bucket(r) for r in short + long_}
    rows_one = Aligner.from_model(model, reference, device="cpu").predict([*reads, *short], batch_size=8).records[:len(reads)]
    rows_other = Aligner.from_model(model, reference, device="cpu").predict([*long_, *reads], batch_size=8).records[len(long_):]
    assert rows_one == rows_other
