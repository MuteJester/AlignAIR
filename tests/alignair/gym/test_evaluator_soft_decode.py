import inspect
from alignair.gym.instrument import evaluator


def test_evaluator_uses_soft_decode():
    src = inspect.getsource(evaluator)
    assert "soft=True" in src, "lattice eval must decode with soft-argmax (spec S3)"
