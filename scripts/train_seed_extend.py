"""Full seed_extend retrain (Gate-1 of the encoder-refactor evaluation): train a production-ish
seed_extend model from scratch on the gym, checkpoint it, and report lattice coord competence +
retrieval recall@k.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/train_seed_extend.py \
      --d-model 96 --steps 8000 --save .private/models/seed_extend_d96.pt
"""
import argparse
import time

import torch
import GenAIRR.data as gdata

from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment_tokens
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.curriculum import StratifiedCurriculum
from alignair.training.gym_trainer import GymTrainer
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.competence import CompetenceMetric
from alignair.gym.instrument.evaluator import LatticeEvaluator
from torch.utils.data import DataLoader

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def retrieval_recall(model, rs, dc, lat, cells, ref_emb, k, n, bs, device):
    out_recall = {}
    for cname in CELLS:
        cur = type("C", (), {"params": lambda s, p=0.0, c=cells[cname]: dict(lat.cell_params(c)),
                             "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
        loader = DataLoader(AlignAIRGym([dc], rs, n=n, seed=0, curriculum=cur),
                            batch_size=bs, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
        hit = tot = 0
        with torch.no_grad():
            for batch in loader:
                batch = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
                o = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
                topk = o["match"]["V"].topk(min(k, o["match"]["V"].shape[-1]), dim=-1).indices
                # set-aware: a hit is the top-k containing ANY allele in the equally-correct
                # GT set (GenAIRR comma-list), not just the first-listed primary — primary-only
                # under-counts recall wherever the read is genuinely ambiguous (heavy SHM).
                mh = batch["v_allele"]                              # (B, n_alleles) multi-hot set
                hit += int((mh.gather(1, topk) > 0).any(dim=1).sum()); tot += mh.shape[0]
        out_recall[cname] = hit / max(tot, 1)
    return out_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--locus", default="igh", choices=["igh", "igk"])
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--coord-tol", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--save", default="")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--reader", action="store_true", help="train the DP log-partition allele reader (set-NCE)")
    ap.add_argument("--reader-weight", type=float, default=1.0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = {"igh": gdata.HUMAN_IGH_OGRDB, "igk": gdata.HUMAN_IGK_OGRDB}[a.locus]
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=a.d_model, n_layers=a.n_layers, nhead=8,
                           dim_feedforward=2 * a.d_model)
    rs = ReferenceSet.from_dataconfigs(dc)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([dc], rs, n=a.batch_size * 8, seed=0, curriculum=StratifiedCurriculum())
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=a.batch_size, device=device,
                         reader=a.reader, reader_weight=a.reader_weight)

    def _save(path):
        if path:
            torch.save({"config": cfg.to_dict(), "model": model.state_dict()}, path)
            print(f"  saved checkpoint -> {path}", flush=True)

    done, t0 = 0, time.perf_counter()
    while done < a.steps:
        chunk = min(a.ckpt_every, a.steps - done)
        trainer.fit(total_steps=chunk, global_total=a.steps, progress=False)
        done += chunk
        el = time.perf_counter() - t0
        print(f"[train] seed_extend {done}/{a.steps}  {el:.0f}s  {el/done:.2f}s/step", flush=True)
        _save(a.save)

    lat = FrozenLattice.standard(seed=0); cells = {c.name: c for c in lat.cells}
    ref_emb = model.encode_reference(rs)
    rec = retrieval_recall(model, rs, dc, lat, cells, ref_emb, a.k, a.eval_n, a.batch_size, device)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(coord_tol=a.coord_tol), [dc], device=device)
    field = ev.eval_all(n_per_cell=a.eval_n)
    print(f"\n=== seed_extend d{a.d_model} {a.steps} steps | competence (tol={a.coord_tol}) + retr@{a.k} ===")
    print(f"{'cell':18s} {'competence':>20s} {'retr@k':>8s}")
    for cname in CELLS:
        v = field.get(cname, {})
        print(f"{cname:18s} {v.get('S', float('nan')):.3f}[{v.get('lo', 0):.3f},{v.get('hi', 0):.3f}]"
              f"   {rec.get(cname, float('nan')):8.3f}")


if __name__ == "__main__":
    main()
