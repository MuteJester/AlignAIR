"""Prediction spread over CPU worker processes (``predict.parallel.ProcessPool``) gives the records one process gives:
the same rows in input order, whatever the number of workers or parts."""
import json

import pytest
import torch

from tests.alignair.predict.golden_rows import FIXTURE, golden_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def aligner():
    from alignair.aligner import Aligner
    model, reference = golden_model()
    return Aligner(model, reference, device="cpu")


def _reads():
    return json.loads(FIXTURE.read_text())["reads"]


def _one_process(aligner, reads, **kwargs):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                         # the workers' thread count: the same float order
    try:
        return aligner.predict(reads, batch_size=4, **kwargs).to_dicts()
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("workers", [2, 3])
def test_workers_give_the_records_one_process_gives_in_input_order(aligner, workers):
    from alignair.predict.parallel import ProcessPool
    reads = _reads() * 3                             # more reads than workers, unevenly split
    with ProcessPool(aligner, workers) as pool:
        records = pool.predict(reads, batch_size=4)
    assert records == _one_process(aligner, reads)


def test_workers_pass_the_prediction_options_on(aligner):
    from alignair.predict.parallel import ProcessPool
    reads = _reads()
    with ProcessPool(aligner, 2) as pool:
        records = pool.predict(reads, batch_size=4, airr=False)
    assert records == _one_process(aligner, reads, airr=False)
    assert "sequence_alignment" not in records[0]


def test_an_empty_chunk_gives_no_records(aligner):
    from alignair.predict.parallel import ProcessPool
    with ProcessPool(aligner, 2) as pool:
        assert pool.predict([], batch_size=4) == []


def test_the_workers_copy_of_the_aligner_is_removed_on_close_and_on_an_error(aligner):
    import os
    from alignair.predict.parallel import ProcessPool
    with ProcessPool(aligner, 2) as pool:
        path = pool._path
        assert os.path.exists(path) and os.stat(path).st_mode & 0o077 == 0     # private to the user
    assert not os.path.exists(path)
    with pytest.raises(RuntimeError):
        with ProcessPool(aligner, 2) as pool:
            path = pool._path
            raise RuntimeError("stop")
    assert not os.path.exists(path)
