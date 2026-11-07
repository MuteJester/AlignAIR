"""Each test starts from the process state the suite set: a test that changes torch's thread count (predict --threads
1 does, in-process) does not hand it to the next test on its worker, whose model outputs depend on it."""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2]

INNER = '''
import pytest
import torch

pytestmark = pytest.mark.component


def test_sets_one_thread():
    torch.set_num_threads(1)


def test_sees_the_budget():
    assert torch.get_num_threads() == 2
'''


def test_a_tests_torch_thread_count_does_not_leak_into_the_next(pytester, monkeypatch):
    """In a subprocess: an in-process run would unload and re-register torch's operator libraries."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
    pytester.makeconftest('pytest_plugins = ("tests.support.environment", "tests.support.policy")')
    pytester.makepyfile(INNER)
    pytester.runpytest_subprocess("-p", "no:randomly", "-p", "no:cacheprovider").assert_outcomes(passed=2)
