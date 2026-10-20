"""Fixtures for the diagnostics tests: small random-init AlignAIR models and the reads, truth and cases of a built
S0 suite."""
from __future__ import annotations

import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402
from alignair.core import AlignAIR  # noqa: E402
from alignair.core.config import AlignAIRConfig  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402

from alignair_assay.core.cases import read_cases, read_fasta  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.reference import dataconfig  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

# Allele temperatures as a released checkpoint carries them, so the replay's calibration step runs.
TEMPERATURES = {"v": 0.5, "d": 0.5, "j": 0.4}
# Heavy-chain segment positions (forward frame, 0-based, end-exclusive) the shaped models predict for every read.
SPANS = {"v": (0, 290), "d": (300, 316), "j": (322, 360)}


class LengthOrientation(torch.nn.Module):
    """An orientation head whose logits pick id (read length mod 4), so reads of varied lengths cover every
    orientation the pipeline canonicalizes."""

    def forward(self, emb, mask):
        return torch.nn.functional.one_hot(mask.sum(1) % 4, 4).float()


class SplitChainType(torch.nn.Module):
    """A chain-type head that puts each batch's reads on two loci, split at the median of one meta-tower feature."""

    def forward(self, meta):
        return torch.nn.functional.one_hot((meta[:, 0] > meta[:, 0].median()).long(), 2).float()


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
def make_shaped_model(make_small_model):
    """``make(variant) -> (model, reference_set)``: a fresh small model whose segment heads put every read's V, D
    and J at the positions of ``SPANS``, so the germline reader and the AIRR assembly work on spans of real size:

    - ``igh``: one locus, no indel predicted;
    - ``igh_varied_orientation``: every orientation id (``LengthOrientation``) and about 4 indels per read;
    - ``igh_igk``: two loci, with the reads of each batch split over them (``SplitChainType``), and about 4 indels.
    """
    def make(variant: str):
        names = ("HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB") if variant == "igh_igk" else ("HUMAN_IGH_OGRDB",)
        model, reference_set = make_small_model(*names, seed=1)
        with torch.no_grad():
            for g, (start, end) in SPANS.items():
                for head, position in ((model.branches[g].start_head, start), (model.branches[g].end_head, end)):
                    head.bias.zero_()
                    head.bias[position] = 30.0
            if variant != "igh":
                model.meta_heads["indel_count"].head.bias.fill_(4.0)
        if variant == "igh_varied_orientation":
            model.orientation_head = LengthOrientation()
        if variant == "igh_igk":
            model.meta_heads["chain_type_logits"] = SplitChainType()
        return model, reference_set

    return make


@pytest.fixture(scope="session")
def shaped_spans():
    """``SPANS``: gene -> (start, end) of every read's segment under ``make_shaped_model``."""
    return dict(SPANS)


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
def s1_layout(tmp_path_factory):
    """A run directory with S1 built on the dev split at scale 0.04, for the tests against the released checkpoint
    (built only when one of them runs)."""
    layout = RunLayout(tmp_path_factory.mktemp("diagnostics_s1"))
    build_suite("S1", layout, reference_name="HUMAN_IGH_OGRDB", split="dev", seed=5, scale=0.04, workers=4)
    return layout


@pytest.fixture(scope="session")
def s0_reads(s0_layout):
    """``(read id, sequence)`` per S0 read, in FASTA order."""
    return read_fasta(s0_layout.reads("S0"))


@pytest.fixture(scope="session")
def s0_truths(s0_layout):
    """The observable truth of every S0 case, in file order."""
    return read_truth_jsonl(s0_layout.truth("S0"))


@pytest.fixture(scope="session")
def s0_cases(s0_layout):
    """Every S0 case (read, GenAIRR record, labels), in file order."""
    return read_cases(s0_layout.cases("S0"))
