"""Fixtures for the diagnostics tests: small random-init AlignAIR models shaped to cover the pipeline's paths (the
plain ones, ``make_small_model`` and ``small_model``, are shared from ``tests/alignair_assay/conftest.py``) and the
reads, truth and cases of a built S0 suite."""
from __future__ import annotations

import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402

from alignair_assay.core.cases import read_cases, read_fasta  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

# Heavy-chain segment positions (forward frame, 0-based, end-exclusive) the shaped models predict for every read,
# and the light-chain ones (no D).
SPANS = {"v": (0, 290), "d": (300, 316), "j": (322, 360)}
LIGHT_SPANS = {"v": (0, 290), "j": (300, 338)}


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
def make_shaped_model(make_small_model):
    """``make(variant) -> (model, reference_set)``: a fresh small model whose segment heads put every read's V, D
    and J at the positions of ``SPANS``, so the germline reader and the AIRR assembly work on spans of real size:

    - ``igh``: one locus, no indel predicted;
    - ``igh_varied_orientation``: every orientation id (``LengthOrientation``) and about 4 indels per read;
    - ``igh_igk``: two loci, with the reads of each batch split over them (``SplitChainType``), and about 4 indels;
    - ``igk``: a light-chain model (no D head; V and J at ``LIGHT_SPANS``) and about 4 indels.
    """
    def make(variant: str):
        names = {"igh_igk": ("HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB"), "igk": ("HUMAN_IGK_OGRDB",)}.get(
            variant, ("HUMAN_IGH_OGRDB",))
        # the varied orientations come from the head alone: no bank vote overrides it
        model, reference_set = make_small_model(*names, seed=1,
                                                bank_orientation=variant != "igh_varied_orientation")
        with torch.no_grad():
            for g, (start, end) in (LIGHT_SPANS if variant == "igk" else SPANS).items():
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


@pytest.fixture(scope="session")
def toy_chip_file(tmp_path_factory):
    """A random-init w64 chip on HUMAN_IGH_OGRDB, saved as an inference model file (the golden records' tests)."""
    import alignair_lab.architectures  # noqa: F401  (registers the lab chips)
    from alignair.core import prepare_model
    from alignair.model_file import save_model
    from alignair.reference.reference_set import ReferenceSet
    import GenAIRR.data as gd
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    torch.manual_seed(0)
    model = prepare_model(reference, architecture="lab-chipw64-v1", boundary_mode="half_open", anchor_heads=True,
                          d_presence_head=True, chip_inputs=True, bank_orientation=True).build_model().eval()
    path = tmp_path_factory.mktemp("golden") / "toy_chip.alignair"
    save_model(str(path), model, dataconfigs=["HUMAN_IGH_OGRDB"], training={}, include_trusted_pickle=False)
    return str(path)


@pytest.fixture(scope="session")
def toy_reads_fasta(tmp_path_factory):
    """Three GenAIRR heavy-chain reads, ``r0`` to ``r2``."""
    import itertools
    import GenAIRR.data as gd
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    component = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)[1]
    reads = [record["sequence"] for record in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, component, 21), 3)]
    path = tmp_path_factory.mktemp("golden_reads") / "reads.fasta"
    path.write_text("".join(f">r{i}\n{read}\n" for i, read in enumerate(reads)))
    return str(path)
