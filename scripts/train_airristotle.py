"""Train AIRRistotle on the GenAIRR gym (MVP: small genotype, forward clean/moderate reads).

Standard modern-LLM training procedure (Llama/Qwen/GPT recipe):
  - AdamW betas=(0.9, 0.95), weight_decay=0.1 applied ONLY to 2-D weights (matmuls/embeddings);
    1-D params (RMSNorm weights, biases) are excluded from decay.
  - LR schedule: linear WARMUP then COSINE decay to a floor (min_lr_ratio * peak).
  - grad-clip 1.0, bf16 autocast.
"""
import argparse, math, os, torch
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle, airristotle_loss
from alignair.airristotle.data import collate
from alignair.airristotle.corpus import ReferenceCorpus, all_dataconfigs, select_configs
from alignair.gym.curriculum import Curriculum


def make_optimizer(model, lr, weight_decay):
    """Standard LLM param grouping: decay 2-D params (matmul weights, embeddings); no decay on
    1-D params (norms, biases)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def lr_at(step, peak, warmup, total, min_ratio):
    """Linear warmup -> cosine decay to min_ratio*peak (the standard LLM schedule)."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=-1, help="-1 -> auto (min(2000, steps//20))")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine floor as a fraction of peak LR")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--v-shortlist", type=int, default=16)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--species", default="all", help="'all' or comma-list, e.g. HUMAN,MOUSE")
    ap.add_argument("--held-out-species", default="", help="comma-list of species excluded from training")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--out", default=".private/models/airristotle_mvp.pt")
    a = ap.parse_args()
    warmup = a.warmup_steps if a.warmup_steps >= 0 else min(2000, max(1, a.steps // 20))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AIRRTokenizer()
    params = dict(Curriculum().params(a.progress))
    cfg = AIRRConfig(vocab_size=tok.vocab_size, v_shortlist=a.v_shortlist)

    held = {n for n in all_dataconfigs()
            if n.split("_")[0] in {s.upper() for s in a.held_out_species.split(",") if s}}
    species = None if a.species == "all" else a.species.split(",")
    configs = select_configs(species=species, exclude=held)
    corpus = ReferenceCorpus(configs, tok, v_shortlist=cfg.v_shortlist)
    m = AIRRistotle(cfg).to(dev).train()
    print(f"AIRRistotle {m.n_params():,} params on {dev} | {len(configs)} references "
          f"(held out {len(held)}) | warmup {warmup} cosine->{a.min_lr_ratio:g}*lr", flush=True)
    opt = make_optimizer(m, a.lr, a.weight_decay)
    gen = corpus.stream(params, n=a.steps * a.batch_size * 2, seed=0)   # ×2 headroom for length skips

    def _save(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model": m.state_dict(), "config": vars(cfg)}, path)

    def next_batch():
        exs = []
        while len(exs) < a.batch_size:
            ex = next(gen)
            if len(ex["input_ids"]) <= cfg.max_seq:          # drop the rare over-long prompt
                exs.append(ex)
        return exs

    for step in range(a.steps):
        exs = next_batch()
        batch = {k: v.to(dev) for k, v in collate(exs, tok.id(tok.PAD)).items()}
        lr = lr_at(step, a.lr, warmup, a.steps, a.min_lr_ratio)
        for pg in opt.param_groups:
            pg["lr"] = lr
        with torch.autocast(dev, dtype=torch.bfloat16):
            loss = airristotle_loss(m(batch["input_ids"]), batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 20 == 0:
            print(f"step {step}  loss {loss.item():.4f}  lr {lr:.2e}  seqlen {batch['input_ids'].shape[1]}", flush=True)
        if a.ckpt_every and step and step % a.ckpt_every == 0:
            _save(a.out)
    _save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
