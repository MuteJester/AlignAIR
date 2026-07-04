"""KV-cache correctness: incremental cached forward must match a full recompute (bit-close)."""
import torch

from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle


def test_cached_incremental_matches_full_forward():
    torch.manual_seed(0)
    cfg = AIRRConfig(vocab_size=15, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                     d_ff=128, max_seq=128)
    m = AIRRistotle(cfg).eval()
    L, p = 24, 16
    ids = torch.randint(0, 15, (1, L))
    with torch.no_grad():
        full = m(ids)                                   # full[:, t] = logits predicting token t+1
        logits, past = m(ids[:, :p], return_past=True)  # prefill prefix
        cached = [logits[:, -1]]                         # == full[:, p-1]
        for t in range(p, L):                            # feed the rest one token at a time
            logits, past = m(ids[:, t:t + 1], past=past, return_past=True)
            cached.append(logits[:, -1])                 # == full[:, t]
    cached = torch.stack(cached, dim=1)                  # (1, L-p+1, V)
    ref = full[:, p - 1:L]
    assert cached.shape == ref.shape
    assert torch.allclose(cached, ref, atol=1e-4), float((cached - ref).abs().max())
