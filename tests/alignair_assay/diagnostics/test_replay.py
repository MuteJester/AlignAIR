"""The stage replay of AlignAIR's predict pipeline: equal to the product, every stage captured, the model's reference
checked against the truth reference, and each substitution fed in between the stages.

The test against the released checkpoint runs only when ``ALIGNAIR_ASSAY_MODEL`` names a ``.alignair`` model file.
"""
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402
from alignair import api  # noqa: E402
from alignair.model_file import save_model  # noqa: E402
from alignair.predict import forward, pipeline, stages  # noqa: E402
from alignair.data import prepare_nucleotide_inputs  # noqa: E402
from alignair.predict.state import GeneCall, GermlineAlignment  # noqa: E402

from alignair_assay.core.cases import read_fasta  # noqa: E402
from alignair_assay.core.reference import Reference  # noqa: E402
from alignair_assay.diagnostics.heads import decode_boundaries  # noqa: E402
from alignair_assay.diagnostics.identity import weights_sha256  # noqa: E402
from alignair_assay.diagnostics.pin import (  # noqa: E402
    PINNED,
    PREDICT_ALIAS,
    code_digest,
    current_digests,
    mirrored,
    pin_changes,
    replay_digest,
    unverified_message,
)
from alignair_assay.diagnostics.replay import (  # noqa: E402
    STAGES,
    StageOutputs,
    Substitutions,
    load_model_checked,
    model_device,
    predict_config,
    replay,
    unwrap_model,
)

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
# Batches of 16 over the 53 inputs: three full batches and a partial one.
BATCH = 16
GENES = ("v", "d", "j")

def _inputs(s0_reads) -> tuple[list[str], list[str]]:
    """The S0 reads plus three that exercise the input policy: an over-length read (cropped to 576), a lower-case
    read and a read with N bases."""
    ids = [name for name, _ in s0_reads]
    seqs = [seq for _, seq in s0_reads]
    extra = {"long": seqs[0] + seqs[1], "lower": seqs[2].lower(), "with_n": seqs[3][:60] + "NNNN" + seqs[3][64:]}
    return ids + list(extra), seqs + list(extra.values())


# --- equivalence with the product -----------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["igh", "igh_varied_orientation", "igk", "unshaped"])
def test_replay_equals_product_pipeline_untrained_model(variant, make_shaped_model, make_small_model, s0_reads):
    model, reference_set = (make_small_model("HUMAN_IGH_OGRDB", seed=0) if variant == "unshaped"
                            else make_shaped_model(variant))
    ids, seqs = _inputs(s0_reads)
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)

    assert out.airr_rows == api.predict_sequences(model, reference_set, seqs, device="cpu", batch_size=BATCH)
    # The light records too: what _to_records gave before the AIRR assembly (which must not have changed them).
    assert out.records == api.predict_sequences(model, reference_set, seqs, device="cpu", batch_size=BATCH,
                                                airr=False)
    # What the variant exercises.
    cropped = [row.get("length_cropped", False) for row in out.airr_rows]
    assert cropped == [i == len(seqs) - 3 for i in range(len(seqs))]
    if variant == "unshaped":                         # the untrained chip as it is: the equality above is the test
        return
    # a read the model puts in a frame without its V holds no V there and has none (predict.v_presence)
    with_v = [row for row in out.airr_rows if row.get("v_presence") != "absent"]
    assert all(row["v_sequence_end"] - row["v_sequence_start"] > 200 for row in with_v)
    assert all(row["airr_assembly_reason"] == "no_v" for row in out.airr_rows if row not in with_v)
    if variant == "igh":
        assert len(with_v) == len(out.airr_rows)      # the bank vote puts every read in its V's frame
    if variant == "igh_varied_orientation":
        assert set(Counter(row["orientation_id"] for row in out.airr_rows)) == {0, 1, 2, 3}
        assert 0 < len(with_v) < len(out.airr_rows)   # the head's frames: some reads keep their V, some do not
    if variant == "igk":
        assert len(with_v) <= len(out.airr_rows) // 2  # an IGK model reads no IGK V in most of the IGH reads
    if variant == "igh_varied_orientation":           # the predicted indels open the gapped germline reader
        assert with_v and any(re.search("[ID]", row["v_cigar"] + row["j_cigar"]) for row in with_v)
    if variant == "igk":
        # A light-chain model: no D head (genes v and j) and the light-chain AIRR assembly.
        assert not model.cfg.has_d and list(out.segments.start) == ["v", "j"]
        assert {row["locus"] for row in out.airr_rows} == {"IGK"}
        assert not any(row.get("d_call") for row in out.airr_rows)


def test_replay_equals_product_pipeline_on_a_multi_locus_reference(make_small_model, s0_reads):
    """IGK + IGL: each row is labelled with its called V's locus, never IGH, by the replay as by the product."""
    model, reference_set = make_small_model("HUMAN_IGK_OGRDB", "HUMAN_IGL_OGRDB", seed=0)
    ids, seqs = _inputs(s0_reads)
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    assert out.airr_rows == api.predict_sequences(model, reference_set, seqs, device="cpu", batch_size=BATCH)
    assert {row["locus"] for row in out.airr_rows} <= {"IGK", "IGL", ""}


def test_replay_equals_product_pipeline_on_a_read_longer_than_the_window(make_small_model, s0_reads):
    from dataclasses import replace
    from alignair.core import AlignAIR
    model, reference_set = make_small_model("HUMAN_IGH_OGRDB", seed=0)
    torch.manual_seed(0)
    model = AlignAIR(replace(model.cfg)).eval()
    model.bind_reference(reference_set)
    ids, seqs = _inputs(s0_reads)
    seqs[-3] = seqs[0] * 3   # longer than the window: cropped to leave the terminal slot
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    assert out.airr_rows == api.predict_sequences(model, reference_set, seqs, device="cpu", batch_size=BATCH)
    assert max(len(read) for read in out.canonical) == model.cfg.max_seq_length - 1


def test_replay_and_product_share_unknown_base_whitespace_and_crop_normalization(small_model):
    from alignair.data import prepare_nucleotide_inputs
    model, reference_set = small_model
    sequences = [' acgr?p ', 'acgt' * 150, ' nN ']
    prepared, _ = prepare_nucleotide_inputs(sequences, model.cfg.max_seq_length - 1)     # the model's window
    out = replay(model, reference_set, sequences, ['one', 'two', 'three'], device='cpu', batch_size=2)
    assert out.airr_rows == api.predict_sequences(model, reference_set, sequences, device='cpu', batch_size=2)
    assert [record['input_sequence'] for record in out.records] == prepared


@needs_model
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("ALIGNAIR_ASSAY_GPU") != "1", reason="set ALIGNAIR_ASSAY_GPU=1"))])
def test_replay_equals_product_pipeline_real_checkpoint(reference, s1_layout, device):
    reads = read_fasta(s1_layout.reads("S1"))[:200]
    ids, seqs = [name for name, _ in reads], [seq for _, seq in reads]
    model, reference_set = load_model_checked(Path(MODEL).expanduser(), reference, device=device)

    out = replay(model, reference_set, seqs, ids, device=device, batch_size=64)
    assert len(out.airr_rows) == 200
    assert out.airr_rows == api.predict_sequences(model, reference_set, seqs, device=device, batch_size=64)


def test_mirrored_product_code_is_pinned():
    # The place that re-pins: when this fails, update diagnostics/replay.py to the new AlignAIR code, run these
    # equivalence tests (with ALIGNAIR_ASSAY_MODEL), then put the new digests in diagnostics/pin.py's PINNED.
    assert api._predict is pipeline.predict  # predict_sequences runs the pipeline the replay mirrors
    assert set(mirrored()) == set(PINNED)
    changed = {name: code_digest(function) for name, function in mirrored().items()
               if code_digest(function) != PINNED[name]}
    assert not changed, (f"AlignAIR code that diagnostics/replay.py mirrors has changed: {sorted(changed)}. Re-check "
                         f"the replay against it, then pin the new digests in diagnostics/pin.py: {changed}")
    assert pin_changes() == {}


def test_pin_changes_name_what_changed(monkeypatch):
    digests = current_digests()
    assert digests[PREDICT_ALIAS] == "alignair.predict.pipeline.predict"
    monkeypatch.setitem(PINNED, "alignair.predict.forward.run_model", "0" * 64)
    assert pin_changes() == {"alignair.predict.forward.run_model": digests["alignair.predict.forward.run_model"]}
    monkeypatch.undo()

    def elsewhere(*args, **kwargs):
        return pipeline.predict(*args, **kwargs)

    monkeypatch.setattr(api, "_predict", elsewhere)
    changes = pin_changes()
    assert list(changes) == [PREDICT_ALIAS] and changes[PREDICT_ALIAS].endswith("elsewhere")
    assert replay_digest() != replay_digest(digests)
    message = unverified_message(changes)
    for part in (PREDICT_ALIAS, changes[PREDICT_ALIAS], "update diagnostics/replay.py", "test_replay.py",
                 "update PINNED", "allow_unverified_replay=True", "--allow-unverified-replay"):
        assert part in message, part


def test_predict_config_is_the_one_predict_sequences_builds(small_model, monkeypatch):
    model, reference_set = small_model
    seen = []
    monkeypatch.setattr(api, "_predict", lambda model, sequences, reference, cfg, device: seen.append(cfg) or [])
    for overrides in ({}, {"threshold": 0.3, "cap": 5, "airr": False}, {"chain_types": ("X",)}):
        api.predict_sequences(model, reference_set, ["ACGT"], device="cpu", batch_size=7, **overrides)
        assert predict_config(model, reference_set, batch_size=7, **overrides) == seen[-1]
    assert seen[0].chain_types == ("IGH",)


# --- captured stages ------------------------------------------------------------------------------------------

def test_stage_outputs_shapes_and_ids(small_model, s0_reads):
    model, reference_set = small_model
    ids, seqs = _inputs(s0_reads)
    n = len(seqs)
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    assert isinstance(out, StageOutputs) and out.ids == ids

    # raw: every forward output, stacked over the batches, equal to the product's batched forward pass.
    cfg = predict_config(model, reference_set, batch_size=BATCH)
    cropped, _ = prepare_nucleotide_inputs(seqs, cfg.max_seq_length)
    batches = forward.run_model(model, cropped, cfg, "cpu")
    assert len(batches) == 4 and set(out.raw) == set(batches[0])
    for key, value in out.raw.items():
        assert np.array_equal(value, np.concatenate([batch[key] for batch in batches])), key
    assert {f"{g}_{side}_logits" for g in GENES for side in ("start", "end")} | {"orientation_logits"} <= set(out.raw)
    assert out.raw["v_start_logits"].shape == (n, 576) and out.raw["orientation_logits"].shape == (n, 4)
    assert out.raw["v_allele"].shape == (n, 198) and out.raw["d_allele"].shape == (n, 33)

    # the predictions' allele probabilities are the forward pass's
    for g in GENES:
        assert np.array_equal(out.predictions.allele[g], out.raw[f"{g}_allele"])
    assert len(out.predictions.orientation) == n

    # one entry per read at every later stage, in input order.
    assert out.canonical == [record["sequence"] for record in out.records]
    assert len(out.canonical[n - 3]) == 575              # the window: the width less the terminal end slot
    for g in GENES:
        assert len(out.segments.start[g]) == len(out.segments.end[g]) == n
        assert len(out.calls[g]) == len(out.germline[g]) == n
        # the record's call is the selector's, unless a presence gate dropped the gene (an empty call)
        assert all(record[f"{g}_call"] == "" or record[f"{g}_call"].split(",")[0] == call.names[0]
                   for call, record in zip(out.calls[g], out.records))
    # a V call is listed with its ties (the call first)
    assert all(record["v_call"] == ",".join(ties)
               for record, ties, gone in zip(out.records, out.v_ties, out.no_v) if not gone)
    assert len(out.segments.low_quality) == n
    assert len(out.records) == len(out.airr_rows) == n
    assert list(out.seconds) == list(STAGES) and all(seconds >= 0 for seconds in out.seconds.values())


def test_allele_names_are_captured(make_shaped_model, s0_reads):
    model, reference_set = make_shaped_model("igh")
    ids, seqs = _inputs(s0_reads)
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    assert out.allele_names == {g: list(reference_set.gene(g.upper()).names) for g in GENES}


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


def test_unknown_decoder_refused():
    with pytest.raises(ValueError, match="decoder 'mean'"):
        Substitutions(decoder="mean")


# --- substitutions ----------------------------------------------------------------------------------------------

def _sub(model, reference_set, seqs, ids, **fields) -> StageOutputs:
    return replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH,
                  substitutions=Substitutions(**fields))


def _same_raw(a: StageOutputs, b: StageOutputs) -> bool:
    return set(a.raw) == set(b.raw) and all(np.array_equal(a.raw[k], b.raw[k]) for k in a.raw)


def test_substitutions_name_what_they_substitute():
    spans = {"v": ([0], [10])}
    assert Substitutions().active() == ()
    assert Substitutions(orientation=[0], spans=spans, calls={"v": [None]},
                         germline={"v": [None]}, decoder="soft").active() == (
        "orientation", "spans", "calls", "germline", "decoder")
    with pytest.raises(ValueError, match="decoder"):
        Substitutions(decoder="other")


def test_orientation_forced_to_the_models_own_ids_is_the_product(make_shaped_model, s0_reads):
    model, reference_set = make_shaped_model("igh_varied_orientation")
    ids, seqs = _inputs(s0_reads)
    base = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    # None keeps the model's own orientation for that read
    own = [int(o) if i % 2 else None for i, o in enumerate(base.predictions.orientation)]
    out = _sub(model, reference_set, seqs, ids, orientation=own)
    assert out.airr_rows == base.airr_rows and _same_raw(out, base)


def test_orientation_is_teacher_forced_and_canonicalizes_the_read(make_shaped_model, s0_reads, s0_truths):
    model, reference_set = make_shaped_model("igh_varied_orientation")
    truth = {t.case_id: t for t in s0_truths}
    ids, seqs = [name for name, _ in s0_reads], [seq for _, seq in s0_reads]
    forced = [truth[name].orientation for name in ids]
    base = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    out = _sub(model, reference_set, seqs, ids, orientation=forced)

    assert list(out.predictions.orientation) == forced
    assert out.canonical == [truth[name].sequence for name in ids]
    assert [row["orientation_id"] for row in out.airr_rows] == forced
    # the orientation head itself is not forced; the rest of the network runs in the forced frame
    assert np.array_equal(out.raw["orientation_logits"], base.raw["orientation_logits"])
    changed = [i for i, o in enumerate(forced) if o != base.predictions.orientation[i]]
    kept = [i for i, o in enumerate(forced) if o == base.predictions.orientation[i]]
    assert changed and kept
    assert not np.allclose(out.raw["v_allele"][changed], base.raw["v_allele"][changed])
    assert np.allclose(out.raw["v_allele"][kept], base.raw["v_allele"][kept])

    with pytest.raises(ValueError, match="orientation id 4"):
        _sub(model, reference_set, seqs, ids, orientation=[4] * len(ids))
    with pytest.raises(ValueError, match="orientation: 2 entries for 50 reads"):
        _sub(model, reference_set, seqs, ids, orientation=[0, 0])


def test_spans_replace_the_corrected_segments(make_shaped_model, s0_reads):
    model, reference_set = make_shaped_model("igh")
    ids, seqs = _inputs(s0_reads)
    n = len(seqs)
    base = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    given = [i for i in range(n) if i % 3 == 0]
    starts = [5 if i in given else None for i in range(n)]
    ends = [200 if i in given else None for i in range(n)]
    out = _sub(model, reference_set, seqs, ids, spans={"v": (starts, ends)})

    seg = out.segments
    assert all(seg.start["v"][i] == 5 and seg.end["v"][i] == 200 for i in given)
    others = [i for i in range(n) if i not in given]
    assert np.array_equal(seg.start["v"][others], base.segments.start["v"][others])
    assert np.array_equal(seg.end["v"][others], base.segments.end["v"][others])
    for g in ("d", "j"):
        assert np.array_equal(seg.start[g], base.segments.start[g])
        assert np.array_equal(seg.end[g], base.segments.end[g])
    # low_quality keeps correct_segments' meaning: a collapsed V or J
    collapsed = ((seg.end["v"] - seg.start["v"]) <= 0) | ((seg.end["j"] - seg.start["j"]) <= 0)
    assert np.array_equal(seg.low_quality, collapsed)
    # the classifier kept its own spans; the germline reader read the substituted ones
    assert _same_raw(out, base)
    assert all(out.germline["v"][i].seq_start >= 5 and out.germline["v"][i].seq_end <= 200 for i in given)
    assert [out.calls[g] for g in GENES] == [base.calls[g] for g in GENES]


@pytest.mark.parametrize("spans, message", [
    ({"v": ([0] * 53, [0] * 53)}, "v span 0..0 of read 0 is empty"),
    ({"v": ([0] * 53, [10_000] * 53)}, "outside read 0"),
    ({"v": ([0] * 52, [10] * 52)}, "spans of v: 52 entries for 53 reads"),
    ({"x": ([0] * 53, [10] * 53)}, "gene 'x'"),
    ({"v": ([0] * 53, [None] * 53)}, "start and end"),
])
def test_bad_spans_refused(spans, message, make_shaped_model, s0_reads):
    model, reference_set = make_shaped_model("igh")
    ids, seqs = _inputs(s0_reads)
    with pytest.raises(ValueError, match=re.escape(message)):
        _sub(model, reference_set, seqs, ids, spans=spans)


def test_calls_replace_the_selected_alleles(make_shaped_model, s0_reads):
    model, reference_set = make_shaped_model("igh")
    ids, seqs = _inputs(s0_reads)
    n = len(seqs)
    base = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    pick = reference_set.gene("V").names[7]
    calls = {"v": [GeneCall((pick,), (1.0,)) if i % 2 == 0 else None for i in range(n)]}
    out = _sub(model, reference_set, seqs, ids, calls=calls)
    for i in range(n):
        if i % 2 == 0:
            assert out.calls["v"][i] == calls["v"][i]
            assert out.records[i]["v_call"].split(",")[0] == pick and out.germline["v"][i].allele == pick
        else:
            assert out.calls["v"][i] == base.calls["v"][i] and out.germline["v"][i] == base.germline["v"][i]
    assert out.calls["j"] == base.calls["j"]

    with pytest.raises(ValueError, match=re.escape("IGHVX9*01 of read 0 is not a v allele of the model")):
        _sub(model, reference_set, seqs, ids, calls={"v": [GeneCall(("IGHVX9*01",), (1.0,))] * n})
    with pytest.raises(ValueError, match="v call of read 0 names no allele"):
        _sub(model, reference_set, seqs, ids, calls={"v": [GeneCall((), ())] * n})


def test_germline_alignments_replace_the_matcher(make_shaped_model, s0_reads, monkeypatch):
    model, reference_set = make_shaped_model("igh")
    ids, seqs = _inputs(s0_reads)
    n = len(seqs)
    base = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    matched = []
    product = stages.align_germline
    monkeypatch.setattr(stages, "align_germline",
                        lambda sequences, *args, **kwargs: matched.append(len(sequences)) or product(sequences, *args,
                                                                                                    **kwargs))
    # The first ten reads get every gene substituted (with the matcher's own alignments, before better_v moves a V to
    # a better allele): the matcher skips them.
    matcher = product(base.canonical, base.segments, base.calls, reference_set,
                      indel_counts=base.predictions.indel_count)
    own = {g: [matcher[g][i] if i < 10 else None for i in range(n)] for g in GENES}
    out = _sub(model, reference_set, seqs, ids, germline=own)
    assert out.airr_rows == base.airr_rows and matched == [n - 10]

    # A substituted alignment, with its call (as a truth substitution gives both), reaches the records and the AIRR
    # assembly.
    v0 = base.calls["v"][0].names[0]
    given = {"v": [GermlineAlignment(v0, 10, 100, 5, 95, "90M")] + [None] * (n - 1)}
    out = _sub(model, reference_set, seqs, ids, germline=given,
               calls={"v": [base.calls["v"][0]] + [None] * (n - 1)})
    record = out.records[0]
    assert (record["v_sequence_start"], record["v_sequence_end"], record["v_germline_start"],
            record["v_germline_end"], record["v_cigar"]) == (10, 100, 5, 95, "90M")
    assert out.records[1:] == base.records[1:]

    other = {"v": [GermlineAlignment("IGHV1-2*02" if v0 != "IGHV1-2*02" else "IGHV1-3*01", 10, 100, 5, 95, "90M")]
             + [None] * (n - 1)}
    with pytest.raises(ValueError, match="is not the read's v call"):
        _sub(model, reference_set, seqs, ids, germline=other)
    outside = {"v": [GermlineAlignment(v0, 10, 10_000, 5, 95, "90M")] + [None] * (n - 1)}
    with pytest.raises(ValueError, match="outside read 0"):
        _sub(model, reference_set, seqs, ids, germline=outside)


def test_the_product_decodes_by_argmax_and_the_soft_decoder_substitutes_the_expectations(small_model, s0_reads):
    model, reference_set = small_model
    ids, seqs = _inputs(s0_reads)
    base = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    soft = _sub(model, reference_set, seqs, ids, decoder="soft")
    lengths = np.array([len(read) for read in base.canonical])
    decoded = {g: decode_boundaries(base, g, "argmax") for g in GENES}
    # the replay's baseline is the product: the heads' argmax decoder is the same implementation
    expected = stages.correct_segments({g: decoded[g][0] for g in GENES}, {g: decoded[g][1] for g in GENES},
                                         lengths, 576, "right")
    expectation = stages.correct_segments({g: base.raw[f"{g}_start"].reshape(-1) for g in GENES},
                                            {g: base.raw[f"{g}_end"].reshape(-1) for g in GENES}, lengths, 576, "right")
    for g in GENES:
        assert np.array_equal(base.segments.start[g], expected.start[g])
        assert np.array_equal(base.segments.end[g], expected.end[g])
        assert np.array_equal(soft.segments.start[g], expectation.start[g])
        assert np.array_equal(soft.segments.end[g], expectation.end[g])
    assert np.array_equal(soft.segments.low_quality, expectation.low_quality)
    assert any(not np.array_equal(soft.segments.start[g], base.segments.start[g]) for g in GENES)
    assert soft.calls == base.calls and _same_raw(soft, base)


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


def test_the_reference_is_checked_before_the_model_moves(make_small_model, reference):
    # Moving to the meta device drops the weights: a model moved before a failed check would be destroyed.
    model, reference_set = make_small_model("HUMAN_IGK_OGRDB", seed=3)
    with pytest.raises(ValueError, match="not the truth reference"):
        load_model_checked((model, reference_set), reference, device="meta")
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    assert model_device(model) == torch.device("cpu")


def test_wrapped_models_are_unwrapped(make_small_model, reference):
    model, reference_set = make_small_model("HUMAN_IGH_OGRDB", seed=4)
    for wrapped in (torch.nn.DataParallel(model), torch.compile(model), torch.nn.DataParallel(torch.compile(model))):
        assert unwrap_model(wrapped) is model
        checked, checked_reference_set = load_model_checked((wrapped, reference_set), reference, device="cpu")
        assert checked is model and checked_reference_set is reference_set
        assert weights_sha256((wrapped, reference_set)) == weights_sha256((model, reference_set))
    assert unwrap_model(model) is model
