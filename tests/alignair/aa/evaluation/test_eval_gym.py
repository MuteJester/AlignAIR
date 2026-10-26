import pytest
import torch

from alignair.aa.eval_gym import finalize_gene, gene_allele_counts

pytestmark = pytest.mark.component


def test_within_class_synonymous_confusion_is_detected():
    # 3 alleles; a0 and a1 are synonymous (same protein class 0), a2 is class 1.
    names = ["a0", "a1", "a2"]
    class_of = {"a0": 0, "a1": 0, "a2": 1}
    # read 0: truth a0; model spreads 0.9/0.8 over the synonymous pair, 0.1 on a2 -> argmax a0 (right nt)
    # read 1: truth a0; model argmax a1 (WRONG nt but SAME class), true a0 still 0.7 (in set)
    probs = torch.tensor([[0.9, 0.8, 0.1],
                          [0.7, 0.95, 0.1]])
    target = torch.tensor([[1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0]])
    c = gene_allele_counts(probs, target, names, class_of, threshold=0.5)
    assert c["n"] == 2
    assert c["nt_top1"] == 1                    # only read 0 gets nt argmax right
    assert c["class_top1"] == 2                 # BOTH get the protein class right
    assert c["errors"] == 1                     # read 1 is an nt error...
    assert c["errors_within_class"] == 1        # ...but a synonymous (within-class) one
    assert c["true_in_predicted_set"] == 2      # true allele > 0.5 in both reads
    assert c["set_size"] == 4                   # read0 {a0,a1}=2, read1 {a0,a1}=2


def test_finalize_computes_the_emergent_ambiguity_signal():
    counts = {"n": 100, "nt_top1": 70, "class_top1": 98, "errors": 30,
              "errors_within_class": 29, "true_in_predicted_set": 96, "set_size": 180}
    r = finalize_gene(counts, has_class=True)
    assert r["nt_top1"] == 0.70 and r["class_top1"] == 0.98
    assert abs(r["frac_errors_within_class"] - 29 / 30) < 1e-9   # ~all nt errors are synonymous
    assert r["true_in_predicted_set"] == 0.96
    assert r["mean_pred_set_size"] == 1.8


def test_unsupervised_rows_are_skipped():
    # an all-zero target row (e.g. inverted D) contributes nothing
    probs = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    c = gene_allele_counts(probs, target, ["a", "b"], None, threshold=0.5)
    assert c["n"] == 1
