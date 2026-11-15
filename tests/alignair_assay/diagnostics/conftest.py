"""Fixtures for the diagnostics tests: small random-init chips shaped to cover the pipeline's paths (the
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


def shape_core(model, spans, indel_count=None) -> None:
    """Make the model's core put every read's segments at ``spans`` (gene -> (start, end)): a forward hook peaks each
    boundary's logits there and sets the soft expectations to it, and with ``indel_count`` sets that output."""
    def hook(module, args, out):
        out = dict(out)
        for g, (start, end) in spans.items():
            for side, position in (("start", start), ("end", end)):
                logits = torch.full_like(out[f"{g}_{side}_logits"], -30.0)
                logits[:, position] = 30.0
                out[f"{g}_{side}_logits"] = logits
                out[f"{g}_{side}"] = torch.full_like(out[f"{g}_{side}"], float(position))
        if indel_count is not None:
            out["indel_count"] = torch.full_like(out["indel_count"], float(indel_count))
        return out
    model.core.register_forward_hook(hook)


@pytest.fixture(scope="session")
def make_shaped_model(make_small_model):
    """``make(variant) -> (model, reference_set)``: a fresh small chip whose core puts every read's V, D and J at the
    positions of ``SPANS`` (``shape_core``), so the germline reader and the AIRR assembly work on spans of real size:

    - ``igh``: no indel predicted;
    - ``igh_varied_orientation``: every orientation id (``LengthOrientation``) and about 4 indels per read;
    - ``igk``: a light-chain model (no D; V and J at ``LIGHT_SPANS``) and about 4 indels.
    """
    def make(variant: str):
        names = ("HUMAN_IGK_OGRDB",) if variant == "igk" else ("HUMAN_IGH_OGRDB",)
        # the varied orientations come from the head alone: no bank vote overrides it
        model, reference_set = make_small_model(*names, seed=1,
                                                bank_orientation=variant != "igh_varied_orientation")
        shape_core(model, LIGHT_SPANS if variant == "igk" else SPANS, indel_count=None if variant == "igh" else 4.0)
        if variant == "igh_varied_orientation":
            model.orientation_head = LengthOrientation()
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
    """A random-init chip (chip-v1, 64 channels) on HUMAN_IGH_OGRDB, saved as an inference model file (the golden
    records' tests)."""
    from alignair.core import prepare_model
    from alignair.model_file import save_model
    from alignair.reference.reference_set import ReferenceSet
    import GenAIRR.data as gd
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    torch.manual_seed(0)
    model = prepare_model(reference).build_model().eval()
    path = tmp_path_factory.mktemp("golden") / "toy_chip.alignair"
    save_model(str(path), model, reference, dataconfigs=["HUMAN_IGH_OGRDB"], training={},
               include_trusted_pickle=False)
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
