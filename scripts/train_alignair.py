"""Train the faithful AlignAIR single-chain model on a GenAIRR dataconfig (retrain from scratch)."""
import argparse

import torch

import GenAIRR.data as gd
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair.core.losses import make_logvars
from alignair.reference.reference_set import ReferenceSet
from alignair.train.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataconfig", nargs="+", default=["HUMAN_IGH_OGRDB"],
                    help="one or more GenAIRR dataconfigs; several => multi-chain (adds chain_type head)")
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--progress", type=float, nargs="+", default=[0.3, 0.6, 0.9],
                    help="difficulty mix: one gym stream per level (round-robin)")
    ap.add_argument("--heavy-shm", type=float, default=0.25,
                    help="add a heavy-SHM stream at this mutation rate (0 to disable)")
    ap.add_argument("--short-boost", type=int, default=1,
                    help="repeat the amplicon (short-read) streams N times to concentrate on short/cropped reads")
    ap.add_argument("--max-len", type=int, default=576)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=".private/models/alignair_single.pt")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--xray-points", type=int, default=None,
                    help="TOTAL number of deep X-ray snapshots over the whole run (overrides --deep-every)")
    ap.add_argument("--deep-every", type=int, default=1000, help="deep X-ray cadence in steps")
    ap.add_argument("--monitor-log", default=None, help="diagnostics JSONL (default: <out>.diag.jsonl)")
    a = ap.parse_args()
    deep_every = max(1, a.steps // a.xray_points) if a.xray_points else a.deep_every

    dcs = [getattr(gd, name) for name in a.dataconfig]
    ref = ReferenceSet.from_dataconfigs(*dcs)
    cfg = AlignAIRConfig.from_dataconfigs(*dcs, max_seq_length=a.max_len)  # counts/has_d/chains auto-derived
    model = AlignAIR(cfg)
    logvars = make_logvars(cfg)
    print(f"train {'+'.join(a.dataconfig)}: V={cfg.v_allele_count} D={cfg.d_allele_count} J={cfg.j_allele_count} "
          f"has_d={cfg.has_d} chains={cfg.num_chain_types} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    if cfg.num_chain_types > 1:
        print("NOTE: multi-chain model built (chain_type head active), but the gym stream trains one "
              "dataconfig at a time; multi-chain data mixing + chain_type targets are a follow-on.", flush=True)
    monitor_log = a.monitor_log or (a.out[:-3] if a.out.endswith(".pt") else a.out) + ".diag.jsonl"
    train(model, ref, dcs[0], cfg, logvars, steps=a.steps, batch_size=a.batch_size, lr=a.lr,
          progresses=tuple(a.progress), heavy_shm=a.heavy_shm, short_boost=a.short_boost, device=a.device,
          save_path=a.out, save_every=a.save_every, resume_path=a.resume, log_every=a.log_every,
          monitor_log=monitor_log, deep_every=deep_every)
    print(f"saved -> {a.out}  (diagnostics: {monitor_log}, deep every {deep_every} "
          f"= {a.steps // deep_every} points)")


if __name__ == "__main__":
    main()
