"""Train AIRRistotle on the GenAIRR gym (MVP: small genotype, forward clean/moderate reads)."""
import argparse, math, torch
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle, airristotle_loss
from alignair.airristotle.data import collate, stream_examples
from alignair.gym.curriculum import Curriculum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-distractors", type=int, default=8)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--out", default=".private/models/airristotle_mvp.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    params = dict(Curriculum().params(a.progress))
    cfg = AIRRConfig(vocab_size=tok.vocab_size)
    m = AIRRistotle(cfg).to(dev).train()
    print(f"AIRRistotle {m.n_params():,} params on {dev}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    gen = stream_examples(rs, tok, params, n=a.steps * a.batch_size, seed=0, n_distractors=a.n_distractors)
    for step in range(a.steps):
        exs = [next(gen) for _ in range(a.batch_size)]
        batch = {k: v.to(dev) for k, v in collate(exs, tok.id(tok.PAD)).items()}
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for pg in opt.param_groups: pg["lr"] = lr
        with torch.autocast(dev, dtype=torch.bfloat16):
            hid, lm = m(batch["input_ids"]); cp = m.copy_logits(hid, hid.shape[1])
            loss = airristotle_loss(lm, cp, batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 20 == 0:
            print(f"step {step}  loss {loss.item():.4f}  lr {lr:.2e}  seqlen {batch['input_ids'].shape[1]}", flush=True)
    import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"model": m.state_dict(), "config": vars(cfg)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
