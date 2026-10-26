
import numpy as np
import pytest
import torch

from alignair.aa.data import collate_aa, record_to_sample
from alignair.aa.equivalence import build_equivalence_classes, db_hash
from alignair.aa.tokenizer import AAProteinTokenizer

pytestmark = pytest.mark.component


def _cmaps():
    v = build_equivalence_classes("V", {"IGHV1-1*01": "QVQLC", "IGHV2-2*01": "EVQLC"}, db_hash([("x", "a")]))
    j = build_equivalence_classes("J", {"IGHJ1*01": "WGQGT", "IGHJ2*01": "WGKGT"}, db_hash([("y", "b")]))
    return {"V": v, "J": j}


def _rec(**over):
    # 24 nt at frame 0 -> exactly 8 residues; sequence_aa MUST be 8 chars to stay in one coord system.
    base = {"sequence": "A" * 24, "sequence_aa": "QVQLCWGQ", "junction_start": 0, "productive": True,
            "v_call": "IGHV1-1*01", "j_call": "IGHJ1*01",
            "v_sequence_start": 0, "v_sequence_end": 12,
            "d_sequence_start": 12, "d_sequence_end": 12,
            "j_sequence_start": 12, "j_sequence_end": 24}
    base.update(over)
    return base


def test_projects_boundaries_and_maps_classes():
    s, why = record_to_sample(_rec(), _cmaps(), AAProteinTokenizer(), 192)
    assert why is None and s is not None
    assert (s["v_start"], s["v_end"]) == (0, 4)      # 4 fully-owned V residues
    assert (s["j_start"], s["j_end"]) == (4, 8)
    assert isinstance(s["tokens"], np.ndarray)


def test_nonzero_frame_offset_shifts_the_residue_grid():
    # junction_start=1 -> frame 1; V=[1,13) J=[13,25); sequence 25nt -> (25-1)//3 = 8 residues.
    s, why = record_to_sample(
        _rec(sequence="A" * 25, junction_start=1, v_sequence_start=1, v_sequence_end=13,
             d_sequence_start=13, d_sequence_end=13, j_sequence_start=13, j_sequence_end=25),
        _cmaps(), AAProteinTokenizer(), 192)
    assert why is None
    assert (s["v_start"], s["v_end"]) == (0, 4) and (s["j_start"], s["j_end"]) == (4, 8)


def test_translation_length_mismatch_is_rejected():
    """A sequence_aa whose length disagrees with (len(sequence)-frame)//3 would put boundaries and
    tokens in different coordinate systems. Verified to hold on real records (0/2965 violations)."""
    s, why = record_to_sample(_rec(sequence_aa="QVQLCWGQG"),      # 9 residues, but 24nt -> 8
                              _cmaps(), AAProteinTokenizer(), 192)
    assert s is None and why == "translation_length_mismatch"


def test_straddling_codon_is_masked_out_of_owned_span():
    # V=[0,11) straddles: residue r3 (nt 9..12) is not fully owned -> V owns only r0..r2.
    s, why = record_to_sample(_rec(v_sequence_end=11), _cmaps(), AAProteinTokenizer(), 192)
    assert why is None and (s["v_start"], s["v_end"]) == (0, 3)


def test_terminal_exclusive_endpoint_equals_residue_count():
    """The genuine terminal case: the exclusive end lands exactly ON the token count, i.e. the
    end slot n that rev-1 could never decode."""
    s, _ = record_to_sample(_rec(), _cmaps(), AAProteinTokenizer(), 192)
    assert len(s["tokens"]) == 8
    assert s["j_end"] == 8 == len(s["tokens"])        # exclusive end == n, not n-1


@pytest.mark.parametrize("over,reason", [
    ({"productive": False}, "not_productive"),
    ({"sequence_aa": ""}, "no_aa"),
    ({"junction_start": None}, "no_frame"),
    ({"sequence_aa": "QVQ*CWGQ"}, "stop_codon"),
    ({"v_call": "IGHV9-9*99"}, "unknown_allele"),
])
def test_rejections_are_reported_by_reason(over, reason):
    s, why = record_to_sample(_rec(**over), _cmaps(), AAProteinTokenizer(), 192)
    assert s is None and why == reason


def test_exact_length_is_rejected_to_reserve_the_endpoint_slot():
    # len(tokens) must be < max_len so slot n exists for an exclusive end.
    s, why = record_to_sample(_rec(sequence="A" * 48, sequence_aa="Q" * 16,
                                   v_sequence_end=24, d_sequence_start=24, d_sequence_end=24,
                                   j_sequence_start=24, j_sequence_end=48),
                              _cmaps(), AAProteinTokenizer(), max_len=16)
    assert s is None and why == "too_long"


def test_collate_shapes_and_dtypes():
    samples = [{"tokens": np.array([1, 2, 3], dtype=np.int64), "v_class": 0, "j_class": 1,
                "v_start": 0, "v_end": 2, "j_start": 2, "j_end": 3},
               {"tokens": np.array([4, 5], dtype=np.int64), "v_class": 1, "j_class": 0,
                "v_start": 0, "v_end": 1, "j_start": 1, "j_end": 2}]
    bi, tg = collate_aa(samples, max_len=192)
    assert bi["tokenized_sequence"].shape == (2, 192)
    assert bi["tokenized_sequence"][1, 2:].sum().item() == 0
    assert tg["v_class"].dtype == torch.long and tg["v_start"].shape == (2, 1)
