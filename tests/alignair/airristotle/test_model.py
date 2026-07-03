"""Pure-LM AIRRistotle: forward shape, masked next-token loss, overfit sanity."""
import torch

from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle, airristotle_loss


def _model(vocab=15):
    cfg = AIRRConfig(vocab_size=vocab, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                     d_ff=128, max_seq=128)
    return AIRRistotle(cfg)


def test_forward_returns_vocab_logits():
    m = _model()
    ids = torch.randint(0, 15, (3, 20))
    logits = m(ids)
    assert logits.shape == (3, 20, 15)


def test_loss_masks_to_output_span():
    m = _model()
    ids = torch.randint(0, 15, (2, 16))
    mask = torch.zeros(2, 16); mask[:, 8:] = 1.0            # only the second half is "output"
    loss = airristotle_loss(m(ids), {"input_ids": ids, "loss_mask": mask})
    assert torch.isfinite(loss) and loss.item() > 0
    # no output tokens -> zero loss
    zero = airristotle_loss(m(ids), {"input_ids": ids, "loss_mask": torch.zeros(2, 16)})
    assert zero.item() == 0.0


def test_overfits_a_fixed_sequence():
    torch.manual_seed(0)
    m = _model()
    ids = torch.randint(1, 15, (4, 24))
    mask = torch.zeros(4, 24); mask[:, 12:] = 1.0
    batch = {"input_ids": ids, "loss_mask": mask}
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for step in range(150):
        opt.zero_grad()
        loss = airristotle_loss(m(ids), batch)
        loss.backward(); opt.step()
        if step == 0:
            first = loss.item()
    assert loss.item() < 0.2 * first, f"did not overfit: {first:.3f} -> {loss.item():.3f}"
    # greedy next-token predictions on the output span match the memorized sequence
    with torch.no_grad():
        pred = m(ids)[:, 11:-1].argmax(-1)
    assert (pred == ids[:, 12:]).float().mean() > 0.95
