"""Prompt/target builder: structure, loss-mask alignment, and copyability of the target."""
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.prompt import build_example, build_prompt


def _fixtures():
    tok = AIRRTokenizer()
    ref = {"V": ["ACGTACGT", "ACGTTTTT"], "D": ["GGGGTTTT"], "J": ["TTTTACGT"]}
    true = {"V": ["ACGTTTTT"], "D": ["GGGGTTTT"], "J": ["TTTTACGT"]}
    return tok, ref, true


def test_example_structure_and_loss_mask():
    tok, ref, true = _fixtures()
    ids, mask, plen = build_example("ACGTTTTTGGGGTTTTTTTTACGT", ref, true, tok)
    assert len(ids) == len(mask)
    assert sum(mask[:plen]) == 0 and all(mask[plen:])          # loss only on target
    toks = tok.decode(ids)
    assert toks[0] == "<REF>" and toks[plen - 1] == "<ALIGN>"  # prompt boundary
    assert toks[plen] == "<V>" and toks[-1] == "<END>"          # target span
    assert "<QUERY>" in toks and "<SEP>" in toks


def test_target_sequences_are_copyable_from_the_prompt():
    """Every true allele must appear verbatim in the prompt (constrained decode requires it)."""
    tok, ref, true = _fixtures()
    ids, _, plen = build_example("ACGT", ref, true, tok)
    prompt_dna = tok.decode_seq(ids[:plen])
    for g in ("V", "D", "J"):
        for s in true[g]:
            assert s in prompt_dna, f"{g} target {s} not copyable from prompt"


def test_absent_gene_emits_none_and_light_chain_omits_d():
    tok, ref, _ = _fixtures()
    # D present in ref but absent in truth -> <NONE> in target
    ids, _, plen = build_example("ACGT", ref, {"V": ["ACGTACGT"], "D": [], "J": ["TTTTACGT"]}, tok)
    assert "<NONE>" in tok.decode(ids[plen:])
    # light chain: no D anywhere
    lc_prompt = build_prompt("ACGT", {"V": ["ACGT"], "J": ["TTTT"]}, tok, has_d=False)
    assert "<D>" not in tok.decode(lc_prompt)
