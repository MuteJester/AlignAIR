"""The stage replay of AlignAIR's predict pipeline: equal to the product, every stage captured, the model's reference
checked against the truth reference.

The test against the released checkpoint runs only when ``ALIGNAIR_ASSAY_MODEL`` names a ``.alignair`` model file.
"""
import hashlib
import inspect
import io
import os
import re
import textwrap
import tokenize
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402
from alignair import api  # noqa: E402
from alignair.model_file import save_model  # noqa: E402
from alignair.predict import forward, pipeline  # noqa: E402
from alignair.predict.calibrate import apply_temperature  # noqa: E402

from alignair_assay.core.cases import read_fasta  # noqa: E402
from alignair_assay.core.reference import Reference  # noqa: E402
from alignair_assay.diagnostics.replay import (  # noqa: E402
    STAGES,
    StageOutputs,
    Substitutions,
    load_model_checked,
    predict_config,
    replay,
)

MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
# Batches of 16 over the 53 inputs: three full batches and a partial one.
BATCH = 16
GENES = ("v", "d", "j")

# The product functions whose orchestration replay.py mirrors (the stage order, the masks, the configuration and
# the batched forward pass), by a digest of their code tokens (comments and blank lines excluded). Any edit to one
# of them fails test_mirrored_product_code_is_pinned: re-check diagnostics/replay.py against the new code, then
# update the digest here.
PINNED = {
    "alignair.predict.pipeline.predict": "ab52fa514e3422438c02121e7a78c729cb7d5371a777e5c52b6c5808c3e09c0d",
    "alignair.api.predict_sequences": "f105197a3fd679737ac5e0cc03e2191350cea8c76fc8f36838f7e7f09c102d7f",
    "alignair.predict.forward.run_model": "5f5a63f21a4224585672f5a7bb8ad2dcc2c3619347b3dbc5bc4d124179a93fc0",
}
MIRRORED = {"alignair.predict.pipeline.predict": pipeline.predict,
            "alignair.api.predict_sequences": api.predict_sequences,
            "alignair.predict.forward.run_model": forward.run_model}


def _code_digest(function) -> str:
    """SHA-256 of the function's tokens as (type name, text), comments and blank lines left out. Token type numbers
    differ between Python versions (3.12 added EXCLAMATION, moving OP from 54 to 55), so the name is hashed. Python
    3.12 also tokenizes f-strings differently (FSTRING_START/MIDDLE/END instead of one STRING), so a pinned function
    that gains an f-string gets a different digest per version; none of them has one."""
    source = textwrap.dedent(inspect.getsource(function))
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    kept = [(tokenize.tok_name[tok.type], tok.string) for tok in tokens
            if tok.type not in (tokenize.COMMENT, tokenize.NL)]
    return hashlib.sha256(repr(kept).encode()).hexdigest()


def _inputs(s0_reads) -> tuple[list[str], list[str]]:
    """The S0 reads plus three that exercise the input policy: an over-length read (cropped to 576), a lower-case
    read and a read with N bases."""
    ids = [name for name, _ in s0_reads]
    seqs = [seq for _, seq in s0_reads]
    extra = {"long": seqs[0] + seqs[1], "lower": seqs[2].lower(), "with_n": seqs[3][:60] + "NNNN" + seqs[3][64:]}
    return ids + list(extra), seqs + list(extra.values())


# --- equivalence with the product -----------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["igh", "igh_varied_orientation", "igh_igk"])
def test_replay_equals_product_pipeline_untrained_model(variant, make_shaped_model, s0_reads):
    model, reference_set = make_shaped_model(variant)
    ids, seqs = _inputs(s0_reads)
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)

    assert out.airr_rows == api.predict_sequences(model, reference_set, seqs, device="cpu", batch_size=BATCH)
    # The light records too: what _to_records gave before the AIRR assembly (which must not have changed them).
    assert out.records == api.predict_sequences(model, reference_set, seqs, device="cpu", batch_size=BATCH,
                                                airr=False)
    # What the variant exercises.
    cropped = [row.get("length_cropped", False) for row in out.airr_rows]
    assert cropped == [i == len(seqs) - 3 for i in range(len(seqs))]
    assert all(row["v_sequence_end"] - row["v_sequence_start"] > 200 for row in out.airr_rows)
    if variant == "igh_varied_orientation":
        assert set(Counter(row["orientation_id"] for row in out.airr_rows)) == {0, 1, 2, 3}
    if variant != "igh":
        assert all(re.search("[ID]", row["v_cigar"] + row["j_cigar"]) for row in out.airr_rows)
    if variant == "igh_igk":
        loci = Counter(row["locus"] for row in out.airr_rows)
        assert set(loci) == {"IGH", "IGK"}
        assert sum(row["d_call"] == "" for row in out.airr_rows) == loci["IGK"]  # no D on a light-chain locus


@needs_model
def test_replay_equals_product_pipeline_real_checkpoint(reference, s1_layout):
    reads = read_fasta(s1_layout.reads("S1"))[:200]
    ids, seqs = [name for name, _ in reads], [seq for _, seq in reads]
    model, reference_set = load_model_checked(Path(MODEL).expanduser(), reference, device="cpu")
    assert model.cfg.allele_temperatures  # the released checkpoint is calibrated: the masks stage does work

    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=64)
    assert len(out.airr_rows) == 200
    assert out.airr_rows == api.predict_sequences(model, reference_set, seqs, device="cpu", batch_size=64)


def test_mirrored_product_code_is_pinned():
    changed = {name: _code_digest(function) for name, function in MIRRORED.items()
               if _code_digest(function) != PINNED[name]}
    assert not changed, (f"AlignAIR code that diagnostics/replay.py mirrors has changed: {sorted(changed)}. Re-check "
                         f"the replay against it, then pin the new digests: {changed}")


def test_predict_config_is_the_one_predict_sequences_builds(small_model, monkeypatch):
    model, reference_set = small_model
    seen = []
    monkeypatch.setattr(api, "_predict", lambda model, sequences, reference, cfg, device: seen.append(cfg) or [])
    for overrides in ({}, {"threshold": 0.3, "cap": 5, "airr": False}, {"chain_types": ("X",)}):
        api.predict_sequences(model, reference_set, ["ACGT"], device="cpu", batch_size=7, **overrides)
        assert predict_config(model, reference_set, batch_size=7, **overrides) == seen[-1]
    assert seen[0].chain_types == ("IGH",) and seen[0].allele_temperatures == model.cfg.allele_temperatures


# --- captured stages ------------------------------------------------------------------------------------------

def test_stage_outputs_shapes_and_ids(small_model, s0_reads):
    model, reference_set = small_model
    ids, seqs = _inputs(s0_reads)
    n = len(seqs)
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    assert isinstance(out, StageOutputs) and out.ids == ids

    # raw: every forward output, stacked over the batches, equal to the product's batched forward pass.
    cfg = predict_config(model, reference_set, batch_size=BATCH)
    cropped, _ = pipeline.apply_input_policy(seqs, cfg.max_seq_length)
    batches = forward.run_model(model, cropped, cfg, "cpu")
    assert len(batches) == 4 and set(out.raw) == set(batches[0])
    for key, value in out.raw.items():
        assert np.array_equal(value, np.concatenate([batch[key] for batch in batches])), key
    assert {f"{g}_{side}_logits" for g in GENES for side in ("start", "end")} | {"orientation_logits"} <= set(out.raw)
    assert out.raw["v_start_logits"].shape == (n, 576) and out.raw["orientation_logits"].shape == (n, 4)
    assert out.raw["v_allele"].shape == (n, 198) and out.raw["d_allele"].shape == (n, 33)

    # predictions are after the calibration: raw holds the probabilities before it.
    for g in GENES:
        assert out.predictions.allele[g].shape == out.raw[f"{g}_allele"].shape
        assert np.array_equal(out.predictions.allele[g],
                              apply_temperature(out.raw[f"{g}_allele"], model.cfg.allele_temperatures[g]))
    assert len(out.predictions.orientation) == n

    # one entry per read at every later stage, in input order.
    assert out.canonical == [record["sequence"] for record in out.records]
    assert len(out.canonical[n - 3]) == 576
    for g in GENES:
        assert len(out.segments.start[g]) == len(out.segments.end[g]) == n
        assert len(out.calls[g]) == len(out.germline[g]) == n
        assert [call.names[0] for call in out.calls[g]] == [record[f"{g}_call"] for record in out.records]
    assert len(out.segments.low_quality) == n
    assert len(out.records) == len(out.airr_rows) == n
    assert list(out.seconds) == list(STAGES) and all(seconds >= 0 for seconds in out.seconds.values())


@pytest.mark.parametrize("variant", ["igh", "igh_igk"])
def test_allele_names_and_locus_masks_are_captured(variant, make_shaped_model, s0_reads):
    model, reference_set = make_shaped_model(variant)
    ids, seqs = _inputs(s0_reads)
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    assert out.allele_names == {g: list(reference_set.gene(g.upper()).names) for g in GENES}
    if variant == "igh":
        assert out.allowed is None  # one locus: the selector runs unmasked
        return
    # Two loci: the per-read masks the selector used, one (reads, alleles) row per read and gene.
    expected = pipeline._locus_allowed(out.predictions.chain_type, reference_set, GENES)
    assert set(out.allowed) == set(GENES)
    for g in GENES:
        assert out.allowed[g].shape == out.predictions.allele[g].shape
        assert np.array_equal(out.allowed[g], expected[g])
        names = out.allele_names[g]
        assert all(set(call.names) <= {names[k] for k in np.flatnonzero(out.allowed[g][i])}
                   for i, call in enumerate(out.calls[g]))
    assert {bool(row.any()) for row in out.allowed["d"]} == {True, False}  # IGK reads may call no D


def test_replay_input_checks(small_model, s0_reads):
    model, reference_set = small_model
    ids, seqs = _inputs(s0_reads)
    with pytest.raises(ValueError, match="2 ids for 3 sequences"):
        replay(model, reference_set, seqs[:3], ids[:2], device="cpu", batch_size=BATCH)
    with pytest.raises(ValueError, match="no sequences"):
        replay(model, reference_set, [], [], device="cpu", batch_size=BATCH)


def test_replay_restores_the_training_mode(small_model, s0_reads):
    model, reference_set = small_model
    ids, seqs = _inputs(s0_reads)
    expected = replay(model, reference_set, seqs[:5], ids[:5], device="cpu", batch_size=BATCH).airr_rows
    model.train()
    try:
        out = replay(model, reference_set, seqs[:5], ids[:5], device="cpu", batch_size=BATCH)
        assert model.training and out.airr_rows == expected  # evaluated in eval mode, then put back
    finally:
        model.eval()


@pytest.mark.parametrize("field", ["orientation", "spans", "classifier_spans", "calls", "germline", "decoder"])
def test_substitutions_are_not_implemented_yet(field, small_model, s0_reads):
    model, reference_set = small_model
    ids, seqs = _inputs(s0_reads)
    value = {"orientation": [0, 0], "spans": {"v": ([0, 0], [10, 10])}, "classifier_spans": True,
             "calls": {"v": [None, None]}, "germline": {"v": [None, None]}, "decoder": "argmax"}[field]
    with pytest.raises(NotImplementedError, match=field):
        replay(model, reference_set, seqs[:2], ids[:2], device="cpu", batch_size=BATCH,
               substitutions=Substitutions(**{field: value}))


def test_unknown_decoder_refused():
    with pytest.raises(ValueError, match="decoder 'mean'"):
        Substitutions(decoder="mean")


# --- the reference check --------------------------------------------------------------------------------------

def test_reference_mismatch_refused_naming_allele(small_model, reference, tmp_path):
    model, reference_set = small_model
    checked = load_model_checked((model, reference_set), reference, device="cpu")
    assert checked[0] is model and checked[1] is reference_set

    v = sorted(reference.alleles["v"])
    changed = Reference({**reference.alleles, "v": {**reference.alleles["v"], v[3]: reference.seq(v[3]) + "A"}}, {})
    with pytest.raises(ValueError) as err:
        load_model_checked((model, reference_set), changed, device="cpu")
    assert f"first difference: {v[3]} has a different sequence in the model reference" in str(err.value)
    assert "it has V=198 D=33 J=7" in str(err.value)

    fewer = Reference({**reference.alleles, "v": {k: s for k, s in reference.alleles["v"].items() if k != v[0]}}, {})
    with pytest.raises(ValueError, match=re.escape(f"{v[0]} is extra in the model reference")):
        load_model_checked((model, reference_set), fewer, device="cpu")
    more = Reference({**reference.alleles, "v": {**reference.alleles["v"], "IGHVX9*01": "ACGT"}}, {})
    with pytest.raises(ValueError) as err:
        load_model_checked((model, reference_set), more, device="cpu")
    assert "IGHVX9*01 is missing from the model reference" in str(err.value)
    assert "the truth reference has V=199 D=33 J=7" in str(err.value)

    # A model file is loaded with alignair's load_model and checked the same way.
    path = tmp_path / "small.alignair"
    save_model(str(path), model, dataconfigs=["HUMAN_IGH_OGRDB"], training={}, include_trusted_pickle=False)
    loaded, loaded_reference_set = load_model_checked(path, reference, device="cpu")
    assert not loaded.training
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), model.state_dict().values()))
    assert Reference.from_reference_set(loaded_reference_set).digest() == reference.digest()
    with pytest.raises(ValueError, match=re.escape(f"{v[3]} has a different sequence in the model reference")):
        load_model_checked(path, changed, device="cpu")
