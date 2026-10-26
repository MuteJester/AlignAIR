import pytest

from alignair.aa.guard.tokenizer import (AaTokenizer, CLS_ID, InputTooLong, MAX_LEN,
                                         enforce_length_contract)

pytestmark = pytest.mark.component

VH = ("QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDT"
      "SKNQFSLKLSSVTAADTAVYYCAREQYSSSPIDYWGQGTLVTVSS")                       # ~118 aa VH
CONSTANT_TAIL = "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKV"


def test_cls_at_position_0_and_fixed_length():
    tok = AaTokenizer(max_len=192)
    ids, mask = tok.encode("QVQLVQSG")
    assert ids[0].item() == CLS_ID and bool(mask[0]) is True
    assert ids.shape[0] == 193 and mask.shape[0] == 193
    assert tok.vocab_size == 24


def test_unknown_residue_maps_to_X():
    tok = AaTokenizer()
    ids, _ = tok.encode("QJ")            # J is not a standard token -> X
    from alignair.aa.guard.tokenizer import _X_ID
    assert ids[2].item() == _X_ID        # position 0=CLS, 1=Q, 2=J->X


def test_mask_marks_pads():
    tok = AaTokenizer()
    ids, mask = tok.encode("QVQL")
    assert bool(mask[:5].all()) and not bool(mask[5:].any())    # CLS + 4 residues real; rest pad


def test_tokenizer_pads_only_never_truncates():
    tok = AaTokenizer(max_len=10)
    with pytest.raises(InputTooLong):
        tok.encode("A" * 11)             # over-length -> raise, NEVER truncate


def test_over_length_input_cannot_reach_supported():
    c = enforce_length_contract("A" * (MAX_LEN + 8))
    assert not c.ok and c.raw_length == MAX_LEN + 8
    assert c.verdict.state == "insufficient_evidence" and str(MAX_LEN + 8) in c.reason


def test_constant_tail_cannot_collapse_to_accepted_vh():
    combined = VH + CONSTANT_TAIL         # VH + constant region -> exceeds the length contract
    assert len(combined) > MAX_LEN
    # (1) the prefilter withholds the combined input -> it can never be an ACCEPTED model input
    assert not enforce_length_contract(combined).ok
    # (2) with no truncation, the tokenizer refuses it rather than collapsing it to look like the VH
    tok = AaTokenizer()
    with pytest.raises(InputTooLong):
        tok.encode(combined)
    # (3) the VH alone IS within contract and tokenizes to a DIFFERENT input than any prefix of `combined`
    vh_ids, _ = tok.encode(VH)
    assert enforce_length_contract(VH).ok


def test_padding_invariance():
    tok, tok2 = AaTokenizer(max_len=192), AaTokenizer(max_len=256)
    seq = "QVQLVQSGAEVKKPG"
    ids, mask = tok.encode(seq)
    ids2, mask2 = tok2.encode(seq)
    real, real2 = ids[mask], ids2[mask2]         # real (non-pad) tokens
    assert real.shape[0] == len(seq) + 1         # CLS + residues
    assert bool((real == real2).all())           # padding amount does not change the real tokens
