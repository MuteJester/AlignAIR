"""Shared fixtures for the assay tests: the frozen GenAIRR, the GenAIRR guard, the truth reference, the committed
bench cases and the small random-init AlignAIR models of the diagnostics and the evaluator (torch and alignair are
imported only by a test that asks for one).

**Frozen GenAIRR.** Before anything imports GenAIRR, the importable package is frozen into
``.private/genairr_frozen/<digest>`` (``core.freeze.freeze_genairr``: reused while GenAIRR is unchanged) and that
root is put first on ``sys.path`` and on ``PYTHONPATH``, so this process, its build workers and the subprocesses it
starts all run one copy while the live checkout is edited. ``ALIGNAIR_ASSAY_GENAIRR_LIVE=1`` runs the live GenAIRR.
The tests that build suites carry the ``genairr_build`` marker. The freeze takes only when nothing imported GenAIRR
before this conftest: true of ``pytest tests/alignair_assay`` (the repository's own plugins import ``alignair``
lazily), not of a whole-repository session, whose product conftests import ``alignair`` (and so GenAIRR) first; that
session runs the GenAIRR they imported, with a warning (``PREIMPORTED`` names it), and leaves ``sys.path`` and
``PYTHONPATH`` as they were: with the copy first there, the processes it starts (spawned training data workers take
this process's ``sys.path``) would import another GenAIRR than this process runs.

**Network.** The repository's isolation fixture forbids every external connection in every test. A test gated by
``ALIGNAIR_ASSAY_NETWORK=1``, which downloads pinned files, asks for ``network``: the real socket functions, taken
when this conftest is imported (outside any test), for that test only.
"""
from __future__ import annotations

import os
import socket
import sys
import warnings
from pathlib import Path

from alignair_assay.core import freeze

# The socket functions the repository's isolation fixture replaces in every test, as they are outside one.
_GETADDRINFO, _CONNECT = socket.getaddrinfo, socket.socket.connect

REPO = Path(__file__).resolve().parents[2]
# The GenAIRR ``__init__`` imported before the freeze from anywhere but the frozen copy (None: the freeze took).
PREIMPORTED = None
if os.environ.get(freeze.LIVE_ENV) != "1":
    FROZEN = freeze.freeze_genairr(REPO / freeze.DEFAULT_DEST)
    loaded = sys.modules.get(freeze.PACKAGE)
    if loaded is not None and Path(loaded.__file__).resolve().parent.parent != FROZEN:
        PREIMPORTED = loaded.__file__
        warnings.warn(f"GenAIRR was imported from {loaded.__file__} before the assay tests froze it: these tests "
                      f"run that copy, not {FROZEN}", stacklevel=1)
    else:
        sys.path.insert(0, str(FROZEN))
        os.environ["PYTHONPATH"] = os.pathsep.join([str(FROZEN), *filter(None, [os.environ.get("PYTHONPATH")])])

import json  # noqa: E402

import pytest  # noqa: E402

from alignair_assay.core.reference import Reference, dataconfig  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE_NAME = "HUMAN_IGH_OGRDB"


def pytest_configure(config):
    config.addinivalue_line("markers", "genairr_build: builds GenAIRR suites (runs the frozen GenAIRR unless "
                                       f"{freeze.LIVE_ENV}=1); -m 'not genairr_build' leaves them out")


@pytest.fixture(scope="session")
def genairr():
    """The GenAIRR package; a test that needs it, directly or through ``reference``, is skipped when it is missing."""
    return pytest.importorskip("GenAIRR")


@pytest.fixture
def network(monkeypatch):
    """Real network access for this test (it is gated by ``ALIGNAIR_ASSAY_NETWORK=1``): the socket functions the
    repository's isolation fixture replaces, restored until the test ends."""
    if os.environ.get("ALIGNAIR_ASSAY_NETWORK") != "1":
        pytest.skip("set ALIGNAIR_ASSAY_NETWORK=1 to download")
    monkeypatch.setattr(socket, "getaddrinfo", _GETADDRINFO)
    monkeypatch.setattr(socket.socket, "connect", _CONNECT)


@pytest.fixture(scope="session")
def genairr_preimported():
    """``PREIMPORTED``: the GenAIRR imported before this conftest could freeze it (None when the freeze took)."""
    return PREIMPORTED


@pytest.fixture(scope="session")
def reference(genairr):
    """The truth reference (``HUMAN_IGH_OGRDB``), built once per test session."""
    return Reference.from_dataconfig(REFERENCE_NAME)


@pytest.fixture(scope="session")
def load_fixture():
    """``load_fixture(name) -> dict``: the committed bench case ``fixtures/<name>``."""
    def load(name: str) -> dict:
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    return load


@pytest.fixture(scope="session")
def make_small_model(genairr):
    """``make(*dataconfig_names, seed=0, bank_orientation=True) -> (model, reference_set)``: a random-init chip
    (embedding 8, an 8-channel trunk) bound to the dataconfigs' reference, in eval mode on the CPU. With ``bank_orientation``
    it orients a read by the reference's k-mers where they decide (``core.config.bank_orientation``), so a read the
    reference reads is in its V's frame as a trained orientation head puts it, and keeps its V call
    (``predict.v_presence``); the random-init head orients the rest."""
    import torch
    from alignair.core import prepare_model
    from alignair.reference.reference_set import ReferenceSet

    def make(*names: str, seed: int = 0, bank_orientation: bool = True):
        reference_set = ReferenceSet.from_dataconfigs(*(dataconfig(name) for name in names))
        setup = prepare_model(reference_set, embed_dim=8, trunk_channels=8, bank_orientation=bank_orientation)
        torch.manual_seed(seed)
        return setup.build_model().eval(), reference_set

    return make


@pytest.fixture(scope="session")
def small_model(make_small_model):
    """The small random-init AlignAIR on ``HUMAN_IGH_OGRDB`` (the truth reference) and its ReferenceSet. Tests
    that change its mode, weights or device must restore them."""
    return make_small_model(REFERENCE_NAME)
