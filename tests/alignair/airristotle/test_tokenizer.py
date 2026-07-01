from alignair.airristotle.tokenizer import AIRRTokenizer


def test_dna_and_specials_roundtrip():
    tok = AIRRTokenizer()
    seq = [tok.GENO, tok.V, "A", "C", "G", "T", "N", tok.S, tok.READ, "G", "A", tok.EOS]
    ids = tok.encode(seq)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == seq


def test_vocab_has_dna_specials_digits():
    tok = AIRRTokenizer()
    for c in "ACGTN0123456789+-":
        assert c in tok.vocab
    for s in (tok.PAD, tok.GENO, tok.READ, tok.ANNOT, tok.V, tok.D, tok.J, tok.S,
              tok.ORI, tok.PROD, tok.EOS, tok.VS, tok.JNE):
        assert s in tok.vocab
    assert tok.id(tok.PAD) == 0
