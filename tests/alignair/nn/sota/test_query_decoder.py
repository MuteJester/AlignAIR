"""Typed VDJ object-query decoder (DETR-style, fixed V/D/J queries)."""
import torch

from alignair.nn.sota.query_decoder import TypedVDJDecoder, GENES


def test_decoder_returns_one_rep_per_gene():
    B, L, d = 4, 30, 32
    mem = torch.randn(B, L, d)
    out = TypedVDJDecoder(d, nhead=4, n_layers=2)(mem)
    assert set(out) == set(GENES)
    for g in GENES:
        assert out[g].shape == (B, d)


def test_typed_queries_are_distinct():
    """V, D, J are three separate learned queries — they must not collapse to the same vector."""
    dec = TypedVDJDecoder(16, nhead=4, n_layers=1)
    q = dec.queries
    assert not torch.allclose(q[0], q[1]) and not torch.allclose(q[1], q[2])


def test_padding_mask_excludes_padded_memory():
    """Padded read positions must not change the decoded queries (masked out of cross-attention)."""
    torch.manual_seed(0)
    B, L, d = 2, 12, 16
    mem = torch.randn(B, L, d)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, 8:] = False                       # last 4 positions are padding
    dec = TypedVDJDecoder(d, nhead=4, n_layers=2).eval()
    with torch.no_grad():
        base = dec(mem, memory_mask=mask)
        mem2 = mem.clone()
        mem2[:, 8:] = torch.randn(B, 4, d)    # scribble on the padded region
        alt = dec(mem2, memory_mask=mask)
    for g in GENES:
        assert torch.allclose(base[g], alt[g], atol=1e-5)


def test_decoder_gradients_flow():
    B, L, d = 2, 10, 16
    mem = torch.randn(B, L, d, requires_grad=True)
    dec = TypedVDJDecoder(d, nhead=4, n_layers=2)
    dec(mem)["V"].sum().backward()
    assert mem.grad is not None and dec.queries.grad is not None
