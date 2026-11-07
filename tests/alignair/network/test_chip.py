"""The chip's core (chip-v1): the trunk, parser, anchors, SHM reader and bank scans wired into the product's outputs, with the
scans' gradient into the parser valved, D presence reading detached evidence, and teacher inputs read only while
training."""
import pytest
import torch

from alignair.core.anchors import ANCHOR_LOGITS
from alignair.core.d_presence import D_PRESENCE_LOGITS
from alignair.errors import ConfigurationError
from alignair.reference.reference_set import ReferenceSet

from alignair.network import chip as chip_core

from .conftest import toy_reference, with_a_novel_v

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
        true = batch.inputs[f"teacher_coord_{gene}"][:, :logits.shape[1]]          # the positions the chip reads
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
        seen.append((bank, torch.equal(tokens, batch.inputs["tokenized_sequence"][:, :tokens.shape[1]])))
        out["v_end_logits"] = out["v_end_logits"] + 1000.
    monkeypatch.setattr(chip_core, "refine_boundaries", lambda *args: None)
    plain = model(batch.inputs)["v_end_logits"]
    monkeypatch.setattr(chip_core, "refine_boundaries", marked)
    refined = model(batch.inputs)["v_end_logits"]
    assert seen == [(model.core.bank, True)]
    read = chip_core.trimmed_width(batch.inputs["tokenized_sequence"] != 0)       # the slots the chip read
    assert torch.allclose(refined[:, :read] - plain[:, :read], torch.full_like(plain[:, :read], 1000.))


def test_the_chip_multiplies_its_d_scan_with_the_local_d_alignment(chip, monkeypatch):
    model, batch, _ = chip
    model.eval()
    leaning = torch.tensor([[0., -50.]])                               # the local alignment's preference: D*01
    monkeypatch.setattr(chip_core, "local_d_log_posterior",
                        lambda out, tokens, mask, bank, alleles: torch.zeros(tokens.shape[0], alleles))
    scan = model(batch.inputs)["d_log_posterior"]
    monkeypatch.setattr(chip_core, "local_d_log_posterior",
                        lambda out, tokens, mask, bank, alleles: leaning.expand(tokens.shape[0], alleles))
    out = model(batch.inputs)
    assert torch.allclose(out["d_log_posterior"], torch.log_softmax(scan + leaning, -1), atol=1e-5)
    assert torch.allclose(out["d_allele"], torch.exp(out["d_log_posterior"] - out["d_log_posterior"].amax(-1, True)))


def test_the_chip_rescores_its_v_call_with_its_bank_at_inference_only(chip, monkeypatch):
    model, batch, _ = chip
    seen = []
    monkeypatch.setattr(chip_core, "rescore_v", lambda out, tokens, bank: seen.append(bank))
    model.train()
    model(batch.inputs)
    assert seen == []
    model.eval()
    model(batch.inputs)
    assert seen == [model.core.bank]


def test_any_reference_binds_and_the_calls_follow_its_alleles(chip):
    """The bank is an input: a reference with an allele the chip never trained on binds, and the V call is over its
    three alleles (no per-allele weights to resize)."""
    model, batch, _ = chip
    model.eval()
    model.bind_reference(with_a_novel_v())
    assert model.cfg.v_allele_count == 3
    inputs = {key: value for key, value in batch.inputs.items() if not key.startswith(("bank_mask", "teacher"))}
    out = model(inputs)
    assert model.core.bank.rows("v") == 3
    assert out["v_allele"].shape[-1] == 3 and out["v_log_posterior"].shape[-1] == 3
    assert out["d_allele"].shape[-1] == 2


def test_the_chip_reads_only_up_to_the_batchs_longest_read_and_gives_the_same_outputs(chip, monkeypatch):
    """The chip computes over the batch's longest read (and its end slot), not the whole window, and pads its outputs
    back: every product output equals the whole window's, and so does every loss part (the per-position internals past
    the reads, which the losses mask, are zeros). Eval mode: dropout and the teacher draws depend on the shapes."""
    from alignair.network.objective import ChipObjective
    model, batch, setup = chip
    model.eval()
    trimmed = model(batch.inputs)
    monkeypatch.setattr(chip_core, "trimmed_width", lambda mask: mask.shape[1])
    whole = model(batch.inputs)
    assert set(trimmed) == set(whole)
    for key, value in whole.items():
        assert trimmed[key].shape == value.shape, key
        if not key.startswith("chip_"):
            assert torch.allclose(trimmed[key], value, atol=1e-5), key
    objective = ChipObjective(setup.config)
    _, trimmed_parts = objective.loss(trimmed, batch.targets)
    _, whole_parts = objective.loss(whole, batch.targets)
    for name, value in whole_parts.items():
        assert torch.allclose(trimmed_parts[name], value, atol=1e-5), name


def test_at_inference_each_genes_columns_are_read_only_where_the_parse_puts_the_gene(chip, monkeypatch):
    """At inference a gene's columns come only from each read's window of bases whose parse marginal for the gene is
    over ``SUPPORT`` (a base outside it would weigh at most that in the gene's scan): the outputs are those of reading
    every base."""
    model, batch, _ = chip
    model.eval()
    inputs = {key: value for key, value in batch.inputs.items() if not key.startswith(("teacher_", "bank_mask_"))}
    calls = []
    top_columns = model.core.reader.top_columns
    monkeypatch.setattr(model.core.reader, "top_columns",
                        lambda features, gene, support, k: calls.append(gene) or top_columns(features, gene, support, k))
    windowed = model(inputs)
    assert sorted(calls) == ["d", "j", "v"]
    monkeypatch.setattr(chip_core, "SUPPORT", -1.)                            # every base, pads included
    every = model(inputs)
    for key, value in every.items():
        assert torch.allclose(windowed[key], value, atol=1e-5), key


def test_a_reference_without_one_of_the_chips_genes_is_refused(chip):
    model, _, _ = chip
    toy = toy_reference()
    light = ReferenceSet({"V": toy.gene("V"), "J": toy.gene("J")}, has_d=False)
    with pytest.raises(ConfigurationError, match="D"):
        model.bind_reference(light)
