"""Explicit execution profiles, reproducible ordering, and required claim evidence."""
import json
import random

import pytest

LAYERS = {"component", "integration", "acceptance"}


def _claim_tests():
    """The release claims' evidence tests: ``tests.support.claims.CLAIM_TESTS``, or a replacement assigned to this
    module's ``CLAIM_TESTS``."""
    claims = globals().get("CLAIM_TESTS")
    if claims is None:
        from .claims import CLAIM_TESTS as claims
    return claims


def __getattr__(name):
    if name == "CLAIM_TESTS":
        return _claim_tests()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def pytest_addoption(parser):
    group = parser.getgroup("alignair QA")
    group.addoption("--model-cache", help="Directory of pre-fetched, checksum-pinned validation models")
    group.addoption("--enforce-claims", action="store_true", help="Require every release claim's evidence to pass")
    group.addoption("--claim-report", help="Write required-claim outcomes as JSON")
    group.addoption("--test-order", choices=("collected", "shuffle", "reverse"), default="collected")
    group.addoption("--order-seed", type=int, default=0, help="Seed used for shuffled test ordering")


def _matches(nodeid, target):
    return nodeid == target or nodeid.startswith(target + "[")


def pytest_configure(config):
    if config.getoption("--enforce-claims") and config.getoption("numprocesses", 0):
        raise pytest.UsageError("Required claim validation must run serially; omit -n")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    config._alignair_claim_results = {}
    config._alignair_selected = []
    for item in items:
        if any(_matches(item.nodeid, target) for target in _claim_tests().values()):
            item.add_marker(pytest.mark.claim_evidence)
        layers = LAYERS & {mark.name for mark in item.iter_markers()}
        if len(layers) != 1:
            raise pytest.UsageError(f"{item.nodeid}: declare exactly one QA layer, found {sorted(layers)}")


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    config = session.config
    config._alignair_selected = [item.nodeid for item in session.items]
    if config.getoption("--test-order") == "shuffle":
        random.Random(config.getoption("--order-seed")).shuffle(session.items)
    elif config.getoption("--test-order") == "reverse":
        session.items.reverse()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    statuses = item.config._alignair_claim_results.setdefault(item.nodeid, {})
    statuses[report.when] = report.outcome


def claim_failures(selected, results):
    failures = {}
    for claim, target in _claim_tests().items():
        matches = [nodeid for nodeid in selected if _matches(nodeid, target)]
        if not matches:
            failures[claim] = {"target": target, "reason": "not selected"}
            continue
        bad = {nodeid: results.get(nodeid, {}) for nodeid in matches
               if results.get(nodeid) != {"setup": "passed", "call": "passed", "teardown": "passed"}}
        if bad:
            failures[claim] = {"target": target, "reason": "required evidence did not pass", "cases": bad}
    return failures


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not config.getoption("--enforce-claims"):
        return
    failures = claim_failures(getattr(config, "_alignair_selected", []),
                              getattr(config, "_alignair_claim_results", {}))
    if config.getoption("--claim-report"):
        from pathlib import Path
        path = Path(config.getoption("--claim-report"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"claims": _claim_tests(), "failures": failures}, indent=2) + "\n")
    if failures:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = config.pluginmanager.getplugin("terminalreporter")
        if reporter:
            reporter.write_sep("=", "Required release-claim evidence is incomplete", red=True)
            for claim, failure in failures.items():
                reporter.write_line(f"{claim}: {failure['reason']} ({failure['target']})", red=True)
