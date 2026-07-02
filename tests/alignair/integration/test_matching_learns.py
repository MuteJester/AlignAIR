import torch
from alignair.data.tokenizer import pad_tokenize, TOKEN_DICT
from alignair.nn.encoder.shared import SharedNucleotideEncoder
from alignair.nn.heads.matching import AlleleMatchingHead, multilabel_match_loss

BASES = "ACGT"


def _rand_seq(rng, n):
    return "".join(BASES[i] for i in rng.integers(0, 4, size=n))


def _noise(rng, seq, p=0.1):
    out = list(seq)
    for i in range(len(out)):
        if rng.random() < p:
            out[i] = BASES[rng.integers(0, 4)]
    return "".join(out)


def test_matching_head_learns_to_identify_alleles():
    import numpy as np
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    K = 12
    refs = [_rand_seq(rng, 80) for _ in range(K)]  # distinct "germline alleles"

    enc = SharedNucleotideEncoder(d_model=64, n_layers=1, nhead=4, max_len=256)
    head = AlleleMatchingHead(init_temp=0.1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)

    ref_tokens, ref_mask = pad_tokenize(refs)
    target = torch.eye(K)

    for _ in range(200):
        E = enc(ref_tokens, ref_mask)                 # (K, d)
        queries = [_noise(rng, r, p=0.1) for r in refs]
        q_tokens, q_mask = pad_tokenize(queries)
        Q = enc(q_tokens, q_mask)                      # (K, d)
        scores = head(Q, E)                            # (K, K)
        loss = multilabel_match_loss(scores, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # after training, a noised query of allele i should retrieve i
    enc.eval()
    with torch.no_grad():
        E = enc(ref_tokens, ref_mask)
        queries = [_noise(rng, r, p=0.1) for r in refs]
        qt, qm = pad_tokenize(queries)
        scores = head(enc(qt, qm), E)
    top1 = (scores.argmax(dim=1) == torch.arange(K)).float().mean().item()
    assert top1 >= 0.8, f"retrieval top-1 too low: {top1}"
