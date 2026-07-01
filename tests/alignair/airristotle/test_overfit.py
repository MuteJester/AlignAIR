import pytest, torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle, airristotle_loss
from alignair.airristotle.data import collate, stream_examples
from alignair.reference.reference_set import ReferenceSet


def test_overfits_tiny_set():
    """The MVP bet: a small model can LEARN to copy calls+coords out of the prompt. Overfit 8 fixed
    examples; the combined loss must fall well below its initial value."""
    torch.manual_seed(0)
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    params = dict(mutation_rate=0.0, productive_only=False, end_loss_5=(0, 0), end_loss_3=(0, 0),
                  indel_count=(0, 0), seq_error_rate=0.0, ambiguous_count=(0, 0))
    exs = list(stream_examples(rs, tok, params, n=8, seed=1, n_distractors=4))
    batch = collate(exs, pad_id=tok.id(tok.PAD))
    cfg = AIRRConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                     d_ff=256, max_seq=batch["input_ids"].shape[1])
    m = AIRRistotle(cfg).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    def step():
        hid, lm = m(batch["input_ids"]); cp = m.copy_logits(hid, hid.shape[1])
        return airristotle_loss(lm, cp, batch)
    l0 = step().item()
    for _ in range(60):
        opt.zero_grad(); loss = step(); loss.backward(); opt.step()
    l1 = step().item()
    assert l1 < 0.5 * l0, (l0, l1)
