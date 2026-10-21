"""Shared fixtures for the assay tests: the frozen GenAIRR, the GenAIRR guard, the truth reference, the committed
bench cases and the small random-init AlignAIR models of the diagnostics and the evaluator (torch and alignair are
imported only by a test that asks for one).

**Frozen GenAIRR.** Before anything imports GenAIRR, the importable package is frozen into
``.private/genairr_frozen/<digest>`` (``core.freeze.freeze_genairr``: reused while GenAIRR is unchanged) and that
root is put first on ``sys.path`` and on ``PYTHONPATH``, so this process, its build workers and the subprocesses it
starts all run one copy while the live checkout is edited. ``ALIGNAIR_ASSAY_GENAIRR_LIVE=1`` runs the live GenAIRR.
The tests that build suites carry the ``genairr_build`` marker.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

from alignair_assay.core import freeze

REPO = Path(__file__).resolve().parents[2]
if os.environ.get(freeze.LIVE_ENV) != "1":
    FROZEN = freeze.freeze_genairr(REPO / freeze.DEFAULT_DEST)
    loaded = sys.modules.get(freeze.PACKAGE)
    if loaded is not None and Path(loaded.__file__).resolve().parent.parent != FROZEN:
        warnings.warn(f"GenAIRR was imported from {loaded.__file__} before the assay tests froze it: these tests "
                      f"run that copy, not {FROZEN}", stacklevel=1)
    sys.path.insert(0, str(FROZEN))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(FROZEN), *filter(None, [os.environ.get("PYTHONPATH")])])

import json  # noqa: E402

import pytest  # noqa: E402

from alignair_assay.core.reference import Reference, dataconfig  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE_NAME = "HUMAN_IGH_OGRDB"
# Allele temperatures as a released checkpoint carries them, so the replay's calibration step runs.
TEMPERATURES = {"v": 0.5, "d": 0.5, "j": 0.4}


def pytest_configure(config):
    config.addinivalue_line("markers", "genairr_build: builds GenAIRR suites (runs the frozen GenAIRR unless "
                                       f"{freeze.LIVE_ENV}=1); -m 'not genairr_build' leaves them out")


@pytest.fixture(scope="session")
def genairr():
    """The GenAIRR package; a test that needs it, directly or through ``reference``, is skipped when it is missing."""
    return pytest.importorskip("GenAIRR")


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
    """``make(*dataconfig_names, seed=0) -> (model, reference_set)``: a random-init AlignAIR with narrow towers
    (embedding 8, 8 filters, 32 tower outputs; about 0.3 M parameters) on the dataconfigs' reference, in eval
    mode on the CPU, with ``TEMPERATURES`` as its allele temperatures."""
    import torch
    from alignair.core import AlignAIR
    from alignair.core.config import AlignAIRConfig
    from alignair.reference.reference_set import ReferenceSet

    def make(*names: str, seed: int = 0):
        configs = [dataconfig(name) for name in names]
        cfg = AlignAIRConfig.from_dataconfigs(*configs, embed_dim=8, filters=8, block_out=32,
                                              allele_temperatures=dict(TEMPERATURES))
        torch.manual_seed(seed)
        return AlignAIR(cfg).eval(), ReferenceSet.from_dataconfigs(*configs)

    return make


@pytest.fixture(scope="session")
def small_model(make_small_model):
    """The small random-init AlignAIR on ``HUMAN_IGH_OGRDB`` (the truth reference) and its ReferenceSet. Tests
    that change its mode, weights or device must restore them."""
    return make_small_model(REFERENCE_NAME)
