"""What the default recipe trains on keeps its bytes: its first 64 records, the two batches the training adapter makes
of them, and the loss and gradient norm of one training step (``tests/support/training_pin.py``). A moved digest is a
moved training batch; Phase 2 keeps every one with novel alleles off."""
import pytest

from tests.support.training_pin import PIN, batch_digest, first_step, pinned_batches, pinned_records, records_digest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def records():
    return pinned_records()


@pytest.fixture(scope="module")
def batches(records):
    return pinned_batches(records)


def test_the_default_recipes_first_records_keep_their_bytes(records):
    assert len(records) == 64 and records_digest(records) == PIN["records_sha256"]


def test_the_default_recipes_first_batches_keep_their_bytes(batches):
    assert [batch_digest(batch) for batch in batches] == PIN["batch_sha256"]
    assert sorted(batches[0].inputs) == ["bank_mask_d", "bank_mask_j", "bank_mask_v", "orientation", "teacher_coord_d",
                                         "teacher_coord_j", "teacher_coord_v", "teacher_region", "tokenized_sequence"]
    assert tuple(batches[0].targets["v_allele"].shape) == (32, 198)


# The step's losses and gradient norm are float32 sums whose order follows the CPU's vector kernels, even on one
# thread: on GitHub's runners the gradient norm moved by two float32 steps (3e-5 of 132) and two losses by one. They
# are pinned to STEP_REL, about eight float32 steps, far below what a changed batch, model or loss moves them.
STEP_REL = 1e-6


def test_a_training_step_on_the_first_batch_keeps_its_loss(batches):
    step, pinned = first_step(batches[0]), PIN["step"]
    assert step.total == pytest.approx(pinned["total"], rel=STEP_REL, abs=0.0)
    assert step.gradient_norm == pytest.approx(pinned["gradient_norm"], rel=STEP_REL, abs=0.0)
    assert sorted(step.parts) == sorted(pinned["parts"])
    assert step.parts == pytest.approx(pinned["parts"], rel=STEP_REL, abs=0.0)
