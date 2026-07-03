"""Constrained decoding: the hard no-hallucination guarantee + output parsing."""
import torch

from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.prompt import build_prompt
from alignair.airristotle.decode import constrained_decode, parse_output


def _tiny_model(vocab):
    cfg = AIRRConfig(vocab_size=vocab, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                     d_ff=128, max_seq=512)
    return AIRRistotle(cfg)


def test_constrained_decode_never_hallucinates_even_untrained():
    torch.manual_seed(0)
    tok = AIRRTokenizer()
    model = _tiny_model(tok.vocab_size)
    ref = {"V": ["ACGTACGTAC", "ACGTTTTTGG", "GGGGCCCCAA"],
           "D": ["GGGGTTTT", "CCCCAAAA"], "J": ["TTTTACGTGG", "GGCACATTAA"]}
    prompt = build_prompt("ACGTTTTTGGGGGGTTTTTTTTACGTGG", ref, tok)
    out = constrained_decode(model, prompt, ref, tok, max_new=300)
    parsed = parse_output(out, tok)
    # EVERY emitted sequence must be an exact reference allele — no hallucination, by construction.
    for G in ("V", "D", "J"):
        for s in parsed.get(G, []):
            assert s in ref[G], f"{G} hallucinated {s!r}"
    assert set(parsed) <= {"V", "D", "J"}


def test_parse_output_roundtrips_structure_incl_ambiguity_and_none():
    tok = AIRRTokenizer()
    ids = ([tok.id(tok.V)] + tok.encode_seq("ACGT") + [tok.id(tok.SEP)] + tok.encode_seq("TTTT")
           + [tok.id(tok.D)] + [tok.id(tok.NONE)]
           + [tok.id(tok.J)] + tok.encode_seq("GGGG") + [tok.id(tok.END)])
    parsed = parse_output(ids, tok)
    assert parsed["V"] == ["ACGT", "TTTT"]        # ambiguity -> two V seqs
    assert parsed["D"] == []                       # <NONE> -> empty
    assert parsed["J"] == ["GGGG"]
