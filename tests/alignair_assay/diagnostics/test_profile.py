"""The speed profile: hooks leave nothing behind, accuracy rows do not depend on it, and the shares add up.

Offline tests run on the small random-init model and a built S0; the test against the released checkpoint runs only
when ``ALIGNAIR_ASSAY_MODEL`` names a ``.alignair`` model file, and its CUDA case when ``ALIGNAIR_ASSAY_GPU=1``.
"""
import os
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.core.cases import read_fasta  # noqa: E402
from alignair_assay.diagnostics.profile import OTHER, hooked_modules, profile_model  # noqa: E402
from alignair_assay.diagnostics.replay import STAGES, load_model_checked, replay  # noqa: E402

BATCH = 16
MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")


def _reads(s0_reads):
    return [sequence for _, sequence in s0_reads], [read_id for read_id, _ in s0_reads]


def _hook_counts(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_profile_cpu_covers_every_module_and_stage(small_model, s0_reads):
    model, reference_set = small_model
    sequences, _ = _reads(s0_reads)
    profile = profile_model(model, reference_set, sequences, device="cpu", batch_size=BATCH, repeats=2)

    assert profile["device"] == "cpu" and profile["batch_size"] == BATCH and profile["reads"] == len(sequences)
    assert profile["reads_per_second"] > 0
    expected = set(hooked_modules(model)) | {OTHER}
    assert set(profile["modules"]) == expected
    assert {"embedding", "orientation_head", "meta_tower", "v.seg_tower", "d.cls_tower", "j.cls_head",
            "meta_heads.mutation_rate"} <= expected
    assert set(profile["stages"]) == set(STAGES)
    for name, entry in profile["modules"].items():
        if name != OTHER:  # the rest of the forward stage can be negligible on a tiny model
            assert entry["ms_per_read"] > 0, name
    assert all(entry["ms_per_read"] > 0 for entry in profile["stages"].values())
    for breakdown in ("modules", "stages"):
        shares = [entry["share"] for entry in profile[breakdown].values()]
        assert sum(shares) == pytest.approx(1.0, abs=1e-6) and all(0 <= s <= 1 for s in shares)


def test_hooks_are_removed_and_accuracy_rows_do_not_depend_on_profiling(small_model, s0_reads):
    model, reference_set = small_model
    sequences, ids = _reads(s0_reads)
    before = replay(model, reference_set, sequences, ids, device="cpu", batch_size=BATCH).airr_rows
    assert _hook_counts(model) == 0
    profile_model(model, reference_set, sequences, device="cpu", batch_size=BATCH, repeats=1)
    assert _hook_counts(model) == 0
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    after = replay(model, reference_set, sequences, ids, device="cpu", batch_size=BATCH).airr_rows
    assert after == before
    assert model.training is False


def test_hooks_are_removed_when_the_run_fails(small_model, s0_reads):
    model, reference_set = small_model
    sequences, _ = _reads(s0_reads)
    original = model.meta_tower.forward

    def broken(*_):
        raise RuntimeError("broken tower")

    model.meta_tower.forward = broken
    try:
        with pytest.raises(RuntimeError, match="broken tower"):
            profile_model(model, reference_set, sequences[:4], device="cpu", batch_size=BATCH, warmup_batches=0,
                          repeats=1)
    finally:
        model.meta_tower.forward = original
    assert _hook_counts(model) == 0


def test_profile_checks_its_inputs(small_model, s0_reads):
    model, reference_set = small_model
    sequences, _ = _reads(s0_reads)
    with pytest.raises(ValueError, match="no sequences"):
        profile_model(model, reference_set, [], device="cpu", batch_size=BATCH)
    with pytest.raises(ValueError, match="batch_size"):
        profile_model(model, reference_set, sequences, device="cpu", batch_size=0)
    with pytest.raises(ValueError, match="repeats"):
        profile_model(model, reference_set, sequences, device="cpu", batch_size=BATCH, repeats=0)


@pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("ALIGNAIR_ASSAY_GPU") != "1", reason="set ALIGNAIR_ASSAY_GPU=1"))])
def test_profile_of_the_released_checkpoint(reference, s1_layout, device):
    model, reference_set = load_model_checked(Path(MODEL).expanduser(), reference, device=device)
    reads = read_fasta(s1_layout.reads("S1"))[:64]
    profile = profile_model(model, reference_set, [s for _, s in reads], device=device, batch_size=32, repeats=2)
    assert profile["reads_per_second"] > 0 and profile["device"] == device
    assert sum(e["share"] for e in profile["modules"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(e["share"] for e in profile["stages"].values()) == pytest.approx(1.0, abs=1e-6)
    assert all(profile["modules"][f"{g}.seg_tower"]["ms_per_read"] > 0 for g in "vdj")
