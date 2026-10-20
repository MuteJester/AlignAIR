"""Fixtures for the diagnostics tests: small random-init AlignAIR models and the reads of a built S0 suite."""
from __future__ import annotations

import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402
from alignair.core import AlignAIR  # noqa: E402
from alignair.core.config import AlignAIRConfig  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402

from alignair_assay.core.cases import read_fasta  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.reference import dataconfig  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

# Allele temperatures as a released checkpoint carries them, so the replay's calibration step runs.
TEMPERATURES = {"v": 0.5, "d": 0.5, "j": 0.4}


@pytest.fixture(scope="session")
def make_small_model():
    """``make(*dataconfig_names, seed=0) -> (model, reference_set)``: a random-init AlignAIR with narrow towers
    (embedding 8, 8 filters, 32 tower outputs; about 0.3 M parameters) on the dataconfigs' reference, in eval
    mode on the CPU, with ``TEMPERATURES`` as its allele temperatures."""
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
    that change its mode or weights must restore them."""
    return make_small_model("HUMAN_IGH_OGRDB")


@pytest.fixture(scope="session")
def s0_layout(tmp_path_factory):
    """A run directory with S0 built on the dev split (50 cases, half of the plain stratum reverse-complemented)."""
    layout = RunLayout(tmp_path_factory.mktemp("diagnostics_s0"))
    build_suite("S0", layout, reference_name="HUMAN_IGH_OGRDB", split="dev", seed=5, scale=1.0, workers=1)
    return layout


@pytest.fixture(scope="session")
def s0_reads(s0_layout):
    """``(read id, sequence)`` per S0 read, in FASTA order."""
    return read_fasta(s0_layout.reads("S0"))
