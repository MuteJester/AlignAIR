"""Train the open-vocabulary VDJ detector on the online GenAIRR gym.

Encoder -> GLIP fusion -> typed V/D/J queries -> decoupled span/objectness/trim + token-level
allele match, supervised by DetectorLoss. A retrieval prefilter (top-k) keeps the full germline
reference tractable. Periodically runs the dynamic-genotype contract eval (canonical / renamed /
novel-SNP) on a held-out batch — the gate that separates "aligns to the reference" from "memorized".

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/train_detector.py \
      --d-model 128 --steps 8000 --batch-size 32 --top-k 32 --save .private/models/detector.pt
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader
import GenAIRR.data as gdata

from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.curriculum import Curriculum
from alignair.nn.sota.detector import OpenVocabVDJDetector
from alignair.nn.sota.loss import DetectorLoss
from alignair.nn.sota.data import CandidateBank, detector_inputs
from alignair.nn.sota.eval import contract_eval


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--encoder-layers", type=int, default=6)
    ap.add_argument("--fusion-layers", type=int, default=2)
    ap.add_argument("--decoder-layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--locus", default="igh", choices=["igh", "igk"])
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--contract-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    dc = {"igh": gdata.HUMAN_IGH_OGRDB, "igk": gdata.HUMAN_IGK_OGRDB}[a.locus]
    rs = ReferenceSet.from_dataconfigs(dc)
    bank = CandidateBank(rs).to(device)

    model = OpenVocabVDJDetector(a.d_model, nhead=a.nhead, encoder_layers=a.encoder_layers,
                                 fusion_layers=a.fusion_layers, decoder_layers=a.decoder_layers,
                                 vocab_size=6).to(device)
    crit = DetectorLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    gym = AlignAIRGym([dc], rs, n=a.batch_size * 8, seed=0, curriculum=Curriculum())
    loader = DataLoader(gym, batch_size=a.batch_size,
                        collate_fn=lambda b: gym_collate(b, rs, rs.has_d))

    n_alleles = {G: bank.sizes[G] for G in bank.genes}
    print(f"[detector] locus={a.locus} d={a.d_model} reference={n_alleles} "
          f"top_k={a.top_k} device={device}", flush=True)

    eval_batch = None
    done, t0 = 0, time.perf_counter()
    model.train()
    while done < a.steps:
        for batch in loader:
            if done >= a.steps:
                break
            batch = _to_device(batch, device)
            if eval_batch is None:
                eval_batch = batch                                  # first batch held out for contract eval
            read_tokens, read_mask, cands, targets = detector_inputs(batch, bank, device=device)
            out = model(read_tokens, read_mask, cands, top_k=a.top_k)
            loss, logs = crit(out, targets)
            if not torch.isfinite(loss):
                done += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()

            gym.set_progress(min(1.0, done / max(a.steps, 1)))       # advance curriculum difficulty
            done += 1

            if done % a.log_every == 0:
                el = time.perf_counter() - t0
                terms = " ".join(f"{k.split('/')[0]}:{v:.2f}" for k, v in logs.items()
                                 if k.endswith("/allele"))
                print(f"[{done}/{a.steps}] loss={logs['total']:.3f}  allele[{terms}]  "
                      f"{el/done:.2f}s/step", flush=True)
            if done % a.contract_every == 0 and eval_batch is not None:
                # full reference (top_k=None): exact name/order invariance + true accuracy ceiling.
                rep = contract_eval(model, rs, eval_batch, top_k=None, n_snps=3, seed=done)
                model.train()
                for cond, acc in rep.items():
                    print(f"    contract/{cond}: " +
                          " ".join(f"{G}={acc[G]:.3f}" for G in acc), flush=True)
            if a.save and done % a.ckpt_every == 0:
                torch.save({"model": model.state_dict(),
                            "args": vars(a), "reference": a.locus}, a.save)
                print(f"    saved -> {a.save}", flush=True)

    if a.save:
        torch.save({"model": model.state_dict(), "args": vars(a), "reference": a.locus}, a.save)
        print(f"[done] saved -> {a.save}", flush=True)


if __name__ == "__main__":
    main()
