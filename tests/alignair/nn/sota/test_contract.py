"""Reference perturbations + the dynamic-genotype contract eval."""
import numpy as np

from alignair.nn.sota.augment import perturb_reference_snps, rename_and_shuffle
from alignair.nn.sota.eval import call_accuracy, contract_eval
from alignair.nn.sota.data import CandidateBank
from alignair.nn.sota.detector import OpenVocabVDJDetector
from alignair.nn.sota.query_decoder import GENES


def _tiny_model():
    return OpenVocabVDJDetector(d_model=32, nhead=4, encoder_layers=2,
                                fusion_layers=1, decoder_layers=2)


def test_perturb_changes_sequences_but_keeps_names_and_order(reference):
    novel = perturb_reference_snps(reference, n_snps=2, rng=np.random.default_rng(0))
    for G in GENES:
        assert novel.gene(G).names == reference.gene(G).names       # order + names preserved
        diffs = [a != b for a, b in zip(novel.gene(G).sequences, reference.gene(G).sequences)]
        assert any(diffs)                                          # at least one allele changed


def test_rename_and_shuffle_remap_recovers_the_true_allele(reference):
    renamed, remap = rename_and_shuffle(reference, rng=np.random.default_rng(1))
    for G in GENES:
        for old in range(len(reference.gene(G).names)):
            new = int(remap[G][old])
            assert renamed.gene(G).sequences[new] == reference.gene(G).sequences[old]


def test_renamed_reference_gives_identical_calls_by_construction(reference, collated):
    """The model ignores names + order, so rename+shuffle must not change any call (the trivial
    half of the contract — proves name/index invariance). Exercised WITH the retrieval prefilter
    (top_k), where an order-dependent forced positive would silently break invariance."""
    model = _tiny_model()
    canon = CandidateBank(reference)
    renamed_ref, remap = rename_and_shuffle(reference, rng=np.random.default_rng(2))
    renamed = CandidateBank(renamed_ref)
    for top_k in (None, 2):                               # 2 forces retrieval on V(4) and D(3)
        assert (call_accuracy(model, canon, collated, top_k=top_k)
                == call_accuracy(model, renamed, collated, top_k=top_k, remap=remap))


def test_contract_eval_reports_all_three_conditions(reference, collated):
    model = _tiny_model()
    report = contract_eval(model, reference, collated, n_snps=2, seed=3)
    assert set(report) == {"canonical", "renamed", "novel_snp2"}
    for cond in report:
        assert set(report[cond]) == set(GENES)
    # untrained model: renamed must still equal canonical (invariance holds before training too)
    assert report["renamed"] == report["canonical"]
