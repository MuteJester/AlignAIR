"""lab-chip-v1's core: the trunk, parser, anchors, SHM reader and bank scans wired into the product's outputs, with the
scans' gradient into the parser valved, D presence reading detached evidence, and teacher inputs read only while
training."""
import pytest
import torch

from alignair.core.anchors import ANCHOR_LOGITS
from alignair.core.d_presence import D_PRESENCE_LOGITS

from alignair_lab.architectures.chip import core as chip_core

pytestmark = pytest.mark.component


PRODUCT_KEYS = {f"{g}_{side}{suffix}" for g in "vdj" for side in ("start", "end") for suffix in ("", "_logits")} | {
    "v_allele", "d_allele", "j_allele", ANCHOR_LOGITS["v"], ANCHOR_LOGITS["j"], D_PRESENCE_LOGITS,
    "mutation_rate", "indel_count"}


def test_forward_emits_the_product_keys(chip):
    model, batch, _ = chip
    out = model.eval()(batch.inputs)
    assert PRODUCT_KEYS <= set(out)
    assert out["v_allele"].shape == (2, 2) and out["v_start"].shape == (2, 1)
    assert out["v_allele"].max(-1).values.allclose(torch.ones(2))         # relative likelihood: the best is 1
    assert out["mutation_rate"].shape == (2, 1) and out["indel_count"].shape == (2, 1)


def test_prediction_batches_get_only_the_product_keys(chip):
    model, batch, _ = chip
    inputs = {key: value for key, value in batch.inputs.items() if not key.startswith(("teacher_", "bank_mask_"))}
    out = model.eval()(inputs)
    assert not any(key.startswith("chip_") for key in out)


def test_eval_mode_ignores_teacher_inputs(chip):
    model, batch, _ = chip
    model.eval()
    plain = model(batch.inputs)
    teacher = dict(batch.inputs)
    teacher["teacher_region"] = torch.zeros_like(teacher["teacher_region"])          # nonsense regions
    with_teacher = model(teacher)
    for key in PRODUCT_KEYS:
        assert torch.equal(plain[key], with_teacher[key]), key


def _true_columns(batch):
    """A stand-in for the reader's columns: each base at its true column (else column 0), probability 1, top-3."""
    def top(logits, k):
        gene = {384: "v", 48: "d", 96: "j"}[logits.shape[-1]]
        true = batch.inputs[f"teacher_coord_{gene}"]
        columns = torch.zeros(*true.shape, k, dtype=torch.long)
        columns[..., 0] = true.clamp_min(0)
        probs = torch.zeros(*true.shape, k)
        probs[..., 0] = (true >= 0).float()
        return columns, probs + 0 * logits[..., :k]
    return top


def test_the_valve_scales_the_scan_gradient_into_the_parser(chip, monkeypatch):
    model, batch, _ = chip
    model.train()
    monkeypatch.setattr(chip_core, "P_TEACHER", 0.0)
    monkeypatch.setattr(chip_core, "top_coordinates", _true_columns(batch))
    gradients = {}
    for scale in (1.0, 0.1):
        monkeypatch.setattr(chip_core, "ROUTE_VALVE", scale)
        model.zero_grad()
        torch.manual_seed(1)
        out = model(batch.inputs)
        out["v_log_posterior"][:, 0].sum().backward()
        gradients[scale] = model.core.parser.emit.weight.grad.clone()
    assert gradients[1.0].abs().sum() > 0
    assert torch.allclose(gradients[0.1], 0.1 * gradients[1.0], rtol=1e-3, atol=1e-7)


def test_d_presence_inputs_are_detached(chip):
    model, batch, _ = chip
    model.train()
    out = model(batch.inputs)
    out[D_PRESENCE_LOGITS].sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in model.core.scans.parameters())
    assert all(p.grad is None or not p.grad.any() for p in model.core.parser.parameters())


def test_bank_mask_restricts_calls_to_the_genotype(chip):
    model, batch, _ = chip
    inputs = dict(batch.inputs)
    inputs["bank_mask_v"] = torch.tensor([[False, True], [False, True]])
    out = model.eval()(inputs)
    assert (out["v_allele"][:, 0] == 0).all() and (out["v_allele"][:, 1] == 1).all()


def test_no_per_allele_parameters(chip):
    model, _, _ = chip
    counts = {2}
    assert not any(set(p.shape) & counts for name, p in model.named_parameters() if "core" in name and p.ndim)
    assert not any("bank" in key for key in model.state_dict())


def test_the_chip_refines_its_v_end_and_j_start_with_its_bank(chip, monkeypatch):
    model, batch, _ = chip
    model.eval()
    seen = []

    def marked(out, tokens, mask, end_mask, bank):
        seen.append((bank, torch.equal(tokens, batch.inputs["tokenized_sequence"])))
        out["v_end_logits"] = out["v_end_logits"] + 1000.
    monkeypatch.setattr(chip_core, "refine_boundaries", lambda *args: None)
    plain = model(batch.inputs)["v_end_logits"]
    monkeypatch.setattr(chip_core, "refine_boundaries", marked)
    refined = model(batch.inputs)["v_end_logits"]
    assert seen == [(model.core.bank, True)]
    assert torch.allclose(refined - plain, torch.full_like(plain, 1000.))
