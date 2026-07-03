"""Multi-reference corpus: config selection + streaming valid examples across references."""
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.corpus import all_dataconfigs, select_configs, ReferenceCorpus
from alignair.gym.curriculum import Curriculum


def test_all_and_select_filters():
    allc = all_dataconfigs()
    assert len(allc) > 50                                    # GenAIRR ships 100+ references
    human = select_configs(species=["human"])
    assert human and all(n.split("_")[0] == "HUMAN" for n in human)
    igh = select_configs(loci=["IGH"])
    assert igh and all(n.split("_")[1] == "IGH" for n in igh)
    first = next(iter(allc))
    assert first not in select_configs(exclude={first})


def test_corpus_streams_valid_examples():
    tok = AIRRTokenizer()
    configs = dict(list(select_configs(species=["human"]).items())[:1])   # one config, fast
    corpus = ReferenceCorpus(configs, tok, v_shortlist=8)
    params = dict(Curriculum().params(0.2))
    exs = list(corpus.stream(params, n=3, seed=0, chunk=3))
    assert len(exs) == 3
    for ex in exs:
        assert len(ex["input_ids"]) == len(ex["loss_mask"])
        assert sum(ex["loss_mask"]) > 0                      # every example has a copy target
