"""The chip's bank scan: every allele row scored by how well its germline explains the read's bases at their predicted
columns, a match costing log(1 - m) and a mismatch log(m / 3)."""
import pytest
import torch

from alignair.data.tokenizer import TOKEN_DICT
from alignair.reference.reference_set import GeneReference, ReferenceSet
from alignair.network.bank import BankCompiler, reverse_complement
from alignair.network.scan import OUT, BankScan, allele_log_posterior

pytestmark = pytest.mark.component

V_ALLELES = ["ACGTACGTAC", "ACGTACGTAA", "ACGTTCGTAC"]     # 1 and 2 differ from 0 at one base each
D_ALLELES = ["GGATCCA", "TTTAAAC"]


def _bank():
    v = GeneReference([f"V*0{i}" for i in range(3)], V_ALLELES, {f"V*0{i}": i for i in range(3)})
    d = GeneReference(["D*01", "D*02"], D_ALLELES, {"D*01": 0, "D*02": 1})
    j = GeneReference(["J*01"], ["TTGG"], {"J*01": 0}, anchors={"J*01": 1})
    return BankCompiler(ReferenceSet({"V": v, "D": d, "J": j}, has_d=True))


def _tokens(sequence):
    return torch.tensor([[TOKEN_DICT[base] for base in sequence]])


def _exact(columns_of_bases):
    """Each base at one column with probability 1 (top-1), as a (1, L, 1) pair."""
    columns = torch.tensor([[[c] for c in columns_of_bases]])
    return columns, torch.ones_like(columns, dtype=torch.float32)


def _score(scan, bank, gene, sequence, columns, m=0.01, weights=None, row_mask=None):
    tokens = _tokens(sequence)
    length = tokens.shape[1]
    columns, probs = columns if isinstance(columns, tuple) else _exact(columns)
    weights = torch.ones(1, length) if weights is None else weights
    rows = bank.rows(gene)
    row_mask = torch.ones(1, rows, dtype=torch.bool) if row_mask is None else row_mask
    return scan(weights, torch.full((1, length), m), tokens, columns, probs,
                getattr(bank, f"table_{gene}"), getattr(bank, f"cover_{gene}"), row_mask)


def test_the_true_row_scores_highest_on_a_clean_read_with_exact_coordinates():
    bank, scan = _bank(), BankScan()
    logits = _score(scan, bank, "v", V_ALLELES[1], list(range(10)))
    assert logits.argmax(-1).item() == 1
    assert logits[0, 1] > logits[0, 0] > float("-inf")


def test_alleles_identical_over_the_covered_columns_tie():
    bank, scan = _bank(), BankScan()
    logits = _score(scan, bank, "v", V_ALLELES[0][:8], list(range(8)))   # alleles 0 and 1 differ at base 9 only
    assert torch.allclose(logits[0, 0], logits[0, 1]) and logits[0, 0] > logits[0, 2]


def test_an_inverted_d_scores_its_reverse_complement_row_highest():
    bank, scan = _bank(), BankScan()
    logits = _score(scan, bank, "d", reverse_complement(D_ALLELES[0]), list(range(7)))
    assert bank.row_inverted_d[logits.argmax(-1).item()] and bank.row_allele_d[logits.argmax(-1).item()] == 0


def test_masked_rows_and_alleles_with_no_allowed_row_are_out():
    bank, scan = _bank(), BankScan()
    logits = _score(scan, bank, "d", D_ALLELES[0], list(range(7)),
                    row_mask=torch.tensor([[False, True, False, True]]))
    assert (logits[0, [0, 2]] <= -1e8).all() and (logits[0, [1, 3]] > -1e8).all()
    posterior = allele_log_posterior(logits, bank.row_allele_d, 2)
    assert posterior[0, 0] <= -1e8 and torch.allclose(posterior[0, 1], torch.tensor(0.), atol=1e-5)


def test_zero_weights_give_finite_uniform_logits():
    bank, scan = _bank(), BankScan()
    logits = _score(scan, bank, "v", V_ALLELES[0], list(range(10)), weights=torch.zeros(1, 10))
    assert torch.isfinite(logits).all() and torch.allclose(logits, logits[:, :1].expand_as(logits))


def test_n_bases_and_zero_weight_bases_do_not_change_scores():
    bank, scan = _bank(), BankScan()
    read = V_ALLELES[2]
    with_n = _score(scan, bank, "v", read[:4] + "N" + read[5:], list(range(10)))
    weights = torch.ones(1, 10)
    weights[0, 4] = 0.
    without = _score(scan, bank, "v", read, list(range(10)), weights=weights)
    assert torch.allclose(with_n, without)


def test_posterior_rows_of_one_allele_combine():
    logits = torch.tensor([[0., 0., 0., 0.]])
    posterior = allele_log_posterior(logits, torch.tensor([0, 1, 0, 1]), 2)
    assert torch.allclose(posterior.exp(), torch.tensor([[0.5, 0.5]]))


def _dense_allele_log_posterior(row_logits, row_allele, n_alleles):
    """The rows-by-alleles definition: an allele's rows' probabilities summed, normalized over the alleles."""
    owner = torch.full((row_logits.shape[1], n_alleles), OUT)
    owner[torch.arange(row_logits.shape[1]), row_allele] = 0.
    allele = torch.logsumexp(row_logits.unsqueeze(-1) + owner, dim=1)
    return (allele - torch.logsumexp(allele, -1, keepdim=True)).masked_fill(allele <= OUT / 2, OUT)


def test_the_allele_posterior_is_the_rows_by_alleles_definition_with_its_gradients():
    """Rows out of the bank, an allele whose rows are all out, an allele with no row and an allele 300 nats behind the
    best: the posterior and its gradients are the dense definition's."""
    torch.manual_seed(0)
    row_allele = torch.tensor([0, 1, 0, 1, 2, 3, 4, 4])                  # allele 5 has no row
    logits = torch.randn(3, 8) * 4
    logits[0, [2, 3]] = OUT
    logits[1, [4]] = OUT                                                  # allele 2's only row is out
    logits[2, [6, 7]] = logits[2].max() - 300.
    upstream = torch.randn(3, 6)
    results = []
    for fn in (_dense_allele_log_posterior, allele_log_posterior):
        rows = logits.clone().requires_grad_(True)
        posterior = fn(rows, row_allele, 6)
        (posterior.masked_fill(posterior <= OUT / 2, 0.) * upstream).sum().backward()
        results.append((posterior.detach(), rows.grad))
    (dense, dense_grad), (posterior, grad) = results
    assert torch.allclose(posterior, dense, atol=1e-4) and torch.allclose(grad, dense_grad, atol=1e-5)
    assert posterior[:, 5].eq(OUT).all() and posterior[1, 2] == OUT and posterior[2, 4] > OUT / 2


def test_gradients_reach_weights_m_and_probs():
    bank, scan = _bank(), BankScan()
    tokens = _tokens(V_ALLELES[0])
    weights = torch.rand(1, 10, requires_grad=True)
    m = torch.full((1, 10), 0.1, requires_grad=True)
    columns = torch.arange(10).reshape(1, 10, 1).expand(1, 10, 3).clone()
    probs = torch.full((1, 10, 3), 0.3, requires_grad=True)
    logits = scan(weights, m, tokens, columns, probs, bank.table_v, bank.cover_v, torch.ones(1, 3, dtype=torch.bool))
    allele_log_posterior(logits, bank.row_allele_v, 3)[0, 0].backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
               for t in (weights, m, probs))
