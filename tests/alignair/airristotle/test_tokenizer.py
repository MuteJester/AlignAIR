"""v2 char-level tokenizer: bases + structure tokens, exact round-trip."""
from alignair.airristotle.tokenizer import AIRRTokenizer


def test_vocab_is_bases_plus_structure_and_pad_is_zero():
    tok = AIRRTokenizer()
    assert tok.id(tok.PAD) == 0
    assert tok.vocab_size == 15                        # 10 specials + 5 bases
    for b in "ACGTN":
        assert b in tok.vocab
    for s in ("<REF>", "<V>", "<D>", "<J>", "<SEP>", "<QUERY>", "<ALIGN>", "<END>", "<NONE>"):
        assert s in tok.vocab


def test_specials_exposed_as_attributes():
    tok = AIRRTokenizer()
    assert tok.V == "<V>" and tok.SEP == "<SEP>" and tok.QUERY == "<QUERY>" and tok.END == "<END>"


def test_encode_seq_and_decode_seq_roundtrip():
    tok = AIRRTokenizer()
    dna = "ACGTTGCAN"
    ids = tok.encode_seq(dna)
    assert tok.decode_seq(ids) == dna
    # unknown char -> N
    assert tok.decode_seq(tok.encode_seq("ACXT")) == "ACNT"


def test_mixed_encode_of_structure_and_bases():
    tok = AIRRTokenizer()
    seq = [tok.V] + list("ACG") + [tok.SEP] + list("TT")
    ids = tok.encode(seq)
    assert tok.decode(ids) == seq
