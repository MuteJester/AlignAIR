"""The default batch size follows where the reads run: 256 on a GPU, 64 in one CPU process, 16 per single-threaded
CPU worker (a worker's activations then stay in its share of the cache: the chip on 8 workers runs 230 reads/s at 16
against 194 at 64)."""
import pytest

from alignair.aligner import auto_batch_size

pytestmark = pytest.mark.component


@pytest.mark.parametrize("device,workers,expected", [("cuda", 1, 256), ("cuda:1", 8, 256), ("cpu", 1, 64),
                                                     ("cpu", 2, 16), ("cpu", 8, 16)])
def test_the_default_batch_size_follows_the_device_and_the_workers(device, workers, expected):
    assert auto_batch_size(device, workers) == expected


def test_one_process_is_the_default():
    assert auto_batch_size("cpu") == 64
