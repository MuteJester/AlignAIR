"""Exercise QA gates through real nested pytest processes."""
import json
import os

import pytest

pytestmark = pytest.mark.integration


def _suite(pytester, monkeypatch, repo_root, *, environment=False):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(repo_root / "src"), str(repo_root))))
    plugins = ["tests.support.policy"]
    if environment:
        plugins.append("tests.support.environment")
    pytester.makeini("""
[pytest]
markers =
    component: test component
    integration: test integration
    acceptance: test acceptance
    claim_evidence: required evidence
    local_data: local data
""")
    pytester.makeconftest(f"""
pytest_plugins = {plugins!r}
from tests.support import policy
policy.CLAIM_TESTS = {{"a public guarantee": "test_contract.py::test_contract"}}
""")


@pytest.mark.parametrize("body,expected", [
    ("assert True", 0),
    ("pytest.skip('missing evidence')", 1),
    ("pytest.xfail('known defect')", 1),
    ("assert False", 1),
])
def test_required_claim_must_actually_pass(pytester, monkeypatch, repo_root, body, expected):
    _suite(pytester, monkeypatch, repo_root)
    pytester.makepyfile(test_contract=f"""
import pytest
pytestmark = pytest.mark.component
def test_contract():
    {body}
""")
    result = pytester.runpytest_subprocess("-q", "--enforce-claims", "--claim-report=claims.json")
    assert result.ret == expected
    report = json.loads((pytester.path / "claims.json").read_text())
    assert bool(report["failures"]) == bool(expected)


def test_missing_claim_cannot_pass_an_empty_selection(pytester, monkeypatch, repo_root):
    _suite(pytester, monkeypatch, repo_root)
    pytester.makepyfile(test_contract="""
import pytest
pytestmark = pytest.mark.component
def test_other():
    assert True
""")
    result = pytester.runpytest_subprocess("-q", "--enforce-claims")
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*not selected*"])


def test_claim_gate_rejects_worker_modes_without_complete_controller_evidence(pytester, monkeypatch, repo_root):
    _suite(pytester, monkeypatch, repo_root)
    pytester.makepyfile(test_contract="""
import pytest
pytestmark = pytest.mark.component
def test_contract():
    assert True
""")
    result = pytester.runpytest_subprocess("-q", "-n", "2", "--enforce-claims")
    assert result.ret == 4
    result.stderr.fnmatch_lines(["*Required claim validation must run serially*"])


def test_required_claim_includes_teardown(pytester, monkeypatch, repo_root):
    _suite(pytester, monkeypatch, repo_root)
    pytester.makepyfile(test_contract="""
import pytest
pytestmark = pytest.mark.component
@pytest.fixture
def resource():
    yield
    raise RuntimeError("teardown failed")
def test_contract(resource):
    assert True
""")
    result = pytester.runpytest_subprocess("-q", "--enforce-claims", "--claim-report=claims.json")
    assert result.ret == 1
    report = json.loads((pytester.path / "claims.json").read_text())
    cases = report["failures"]["a public guarantee"]["cases"]
    assert next(iter(cases.values()))["teardown"] == "failed"


def test_random_streams_are_independent_of_test_order(pytester, monkeypatch, repo_root):
    _suite(pytester, monkeypatch, repo_root, environment=True)
    pytester.makepyfile(test_contract="""
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
pytestmark = pytest.mark.component
def draw(name):
    path = Path("draws.json")
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = [random.random(), float(np.random.rand()), float(torch.rand(()))]
    path.write_text(json.dumps(data))
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
def test_contract():
    draw("first")
def test_other():
    draw("second")
""")
    normal = pytester.runpytest_subprocess("-q")
    assert normal.ret == 0
    path = pytester.path / "draws.json"
    expected = json.loads(path.read_text())
    path.unlink()
    reversed_run = pytester.runpytest_subprocess("-q", "--test-order=reverse")
    assert reversed_run.ret == 0
    assert json.loads(path.read_text()) == expected


def test_external_network_is_rejected_before_connecting(pytester, monkeypatch, repo_root):
    _suite(pytester, monkeypatch, repo_root, environment=True)
    pytester.makepyfile(test_contract="""
import socket
import pytest
pytestmark = pytest.mark.component
def test_contract():
    socket.create_connection(("203.0.113.1", 443), timeout=1)
""")
    result = pytester.runpytest_subprocess("-q")
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*External DNS lookup is forbidden*"])
