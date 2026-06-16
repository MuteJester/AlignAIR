import numpy as np
import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.backbone import SequenceBackbone
from alignair.nn.region_head import RegionTagger, decode_boundaries, REGION_INDEX

# dominant base per region (others sampled uniformly with low prob) -> learnable composition
DOM = {"pre": "T", "V": "A", "N1": "N", "D": "G", "N2": "N", "J": "C", "post": "T"}


def _gen(rng):
    layout = [("pre", rng.integers(2, 6)), ("V", rng.integers(40, 56)),
              ("N1", rng.integers(2, 6)), ("D", rng.integers(8, 14)),
              ("N2", rng.integers(2, 6)), ("J", rng.integers(20, 28)),
              ("post", rng.integers(2, 6))]
    seq, labels = [], []
    for name, n in layout:
        dom = DOM[name]
        for _ in range(int(n)):
            base = dom if rng.random() < 0.7 else "ACGT"[rng.integers(0, 4)]
            seq.append(base)
            labels.append(REGION_INDEX[name])
    return "".join(seq), labels


def _batch(rng, B):
    seqs, lab = zip(*[_gen(rng) for _ in range(B)])
    tokens, mask = pad_tokenize(list(seqs))
    L = tokens.shape[1]
    label_t = torch.zeros(B, L, dtype=torch.long)  # 0 = pad
    for i, ls in enumerate(lab):
        label_t[i, :len(ls)] = torch.tensor(ls)
    return tokens, mask, label_t


def test_region_tagging_recovers_exact_boundaries():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    bb = SequenceBackbone(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    tagger = RegionTagger(d_model=64)
    opt = torch.optim.Adam(list(bb.parameters()) + list(tagger.parameters()), lr=2e-3)
    ce = torch.nn.CrossEntropyLoss()

    for _ in range(120):
        tokens, mask, labels = _batch(rng, 16)
        logits = tagger(bb(tokens, mask))
        loss = ce(logits[mask], labels[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()

    # evaluate exact-boundary deviation on a fresh batch
    bb.eval()
    tagger.eval()
    with torch.no_grad():
        tokens, mask, labels = _batch(rng, 32)
        logits = tagger(bb(tokens, mask))
    pred = decode_boundaries(logits, mask, has_d=True)
    devs = []
    for b in range(32):
        for g in ("V", "D", "J"):
            gid = REGION_INDEX[g]
            gt = (labels[b] == gid).nonzero(as_tuple=True)[0]
            gs, ge = int(gt.min()), int(gt.max()) + 1
            devs.append(abs(pred[b][f"{g.lower()}_start"] - gs))
            devs.append(abs(pred[b][f"{g.lower()}_end"] - ge))
    mean_dev = float(np.mean(devs))
    assert mean_dev <= 2.0, f"mean boundary deviation too high: {mean_dev}"
